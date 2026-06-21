import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Any, Dict, List
from functools import partial


def _get_shard(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shard a tensor evenly along dimension 0."""
    total_size = tensor.size(0)
    shard_size = total_size // world_size
    start = rank * shard_size
    end = start + shard_size
    return tensor[start:end].contiguous()


class FSDP(nn.Module):
    """
    Fully-Sharded Data Parallel implementation.
    
    Wraps an nn.Module and shards all Linear/Embedding layer parameters.
    Uses forward/backward hooks for all-gather and reduce-scatter.
    """
    
    def __init__(self, module: nn.Module, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized")
            
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        
        # Store module as a proper submodule so nn.Module tracks it
        self.module = module
        
        # Track parameter states: id(sharded_param) -> metadata dict
        self._param_info: Dict[int, Dict] = {}
        
        # Ordered layers for prefetch
        self._layers: List[nn.Module] = []
        self._layer_params: Dict[int, List[int]] = {}  # id(layer) -> [param_ids]
        
        # Track non-FSDP (replicated) parameters that need gradient all-reduce
        self._replicated_params: List[nn.Parameter] = []
        
        # Async communication handles
        self._handles: List[Any] = []
        
        self._setup_sharding()
        self._register_hooks()
        
    def _setup_sharding(self):
        """Find all Linear/Embedding layers and shard their parameters."""
        layer_idx = 0
        
        # First pass: identify all Linear/Embedding layers
        fsdp_modules = set()
        for name, m in self.module.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                fsdp_modules.add(id(m))
        
        # Second pass: shard FSDP layers, track replicated params
        for name, m in self.module.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                self._layers.append(m)
                self._layer_params[id(m)] = []
                
                for p_name in ['weight', 'bias']:
                    if not hasattr(m, p_name):
                        continue
                    p = getattr(m, p_name)
                    if p is None or not isinstance(p, nn.Parameter):
                        continue
                        
                    # Create shard along dim 0
                    shard = _get_shard(p.data, self.rank, self.world_size)
                    sharded_p = nn.Parameter(shard, requires_grad=p.requires_grad)
                    
                    # Store metadata
                    info = {
                        'sharded': sharded_p,
                        'original_shape': p.shape,
                        'original_device': p.device,
                        'module': m,
                        'param_name': p_name,
                        'layer_idx': layer_idx,
                        'gathered': None,
                    }
                    self._param_info[id(sharded_p)] = info
                    self._layer_params[id(m)].append(id(sharded_p))
                    
                    # Replace parameter in the layer
                    setattr(m, p_name, sharded_p)
                    
                layer_idx += 1
        
        # Track replicated (non-FSDP) parameters for gradient all-reduce
        for name, p in self.module.named_parameters():
            if id(p) not in self._param_info and p.requires_grad:
                self._replicated_params.append(p)
                
    def _register_hooks(self):
        """Register forward hooks on all managed layers."""
        for idx, layer in enumerate(self._layers):
            # Pre-forward: gather current layer, prefetch layer+2
            layer.register_forward_pre_hook(
                lambda m, inp, idx=idx: self._pre_forward(m, inp, idx)
            )
            # Post-forward: free memory
            layer.register_forward_hook(
                lambda m, inp, out, idx=idx: self._post_forward(m, inp, out, idx)
            )
            
    def _pre_forward(self, module, inp, idx):
        """All-gather weights before forward computation."""
        # Wait for pending async operations
        self._wait_handles()
        
        # Gather current layer
        for pid in self._layer_params[id(module)]:
            self._gather_param(pid)
            
        # Prefetch layer idx+2
        prefetch_idx = idx + 2
        if prefetch_idx < len(self._layers):
            prefetch_layer = self._layers[prefetch_idx]
            for pid in self._layer_params[id(prefetch_layer)]:
                self._gather_param(pid, async_op=True)
                
    def _post_forward(self, module, inp, out, idx):
        """After forward, free gathered weights that are no longer needed."""
        # Keep current, next, and prefetch layers
        keep_indices = {idx, idx + 1, idx + 2}
        for i, layer in enumerate(self._layers):
            if i in keep_indices:
                continue
            for pid in self._layer_params[id(layer)]:
                info = self._param_info[pid]
                if info['gathered'] is not None:
                    # Restore sharded parameter
                    setattr(info['module'], info['param_name'], info['sharded'])
                    info['gathered'] = None
                    
    def _gather_param(self, pid: int, async_op: bool = False):
        """All-gather a sharded parameter to reconstruct the full tensor."""
        info = self._param_info[pid]
        if info['gathered'] is not None:
            return
            
        sharded = info['sharded']
        
        # All-gather: create output list and gather
        output_tensors = [torch.empty_like(sharded.data) for _ in range(self.world_size)]
        input_tensor = sharded.data
        
        if async_op:
            handle = dist.all_gather(output_tensors, input_tensor, async_op=True)
            self._handles.append(handle)
            handle.wait()
            self._handles.remove(handle)
        else:
            dist.all_gather(output_tensors, input_tensor)
        
        # Concatenate along dim 0
        full = torch.cat(output_tensors, dim=0)
        
        # Cast to compute_dtype if specified
        if self.compute_dtype is not None and full.dtype != self.compute_dtype:
            full = full.to(self.compute_dtype)
        
        # CRITICAL FIX: Detach to break computation graph with all_gather
        # This prevents autograd from automatically computing reduce_scatter through all_gather's backward
        full = full.detach().requires_grad_(sharded.requires_grad)
        
        # Register hook to reduce-scatter gradients after backward
        if sharded.requires_grad:
            full.register_hook(partial(self._post_backward_hook, pid=pid))
            
        # Store gathered tensor and temporarily install in module
        info['gathered'] = full
        setattr(info['module'], info['param_name'], full)
        
    def _post_backward_hook(self, grad, pid):
        """Reduce-scatter gradient after it is computed for a gathered param."""
        info = self._param_info[pid]
        sharded = info['sharded']
        
        # Cast gradient back to fp32 if compute_dtype was used
        if self.compute_dtype is not None:
            grad = grad.to(torch.float32)
        
        # Reduce-scatter to the appropriate rank in FP32 for numerical stability
        grad_chunks = list(grad.chunk(self.world_size, dim=0))
        grad_shard = torch.empty(sharded.data.shape, dtype=torch.float32, device=sharded.data.device)
        dist.reduce_scatter(grad_shard, grad_chunks, op=dist.ReduceOp.SUM)
        
        # Average so that the update matches single-GPU training
        grad_shard = grad_shard / self.world_size
        
        # Cast back to master weight dtype (FP32)
        grad_shard = grad_shard.to(sharded.data.dtype)
        
        # Accumulate into sharded parameter gradient
        if sharded.grad is None:
            sharded.grad = grad_shard
        else:
            sharded.grad += grad_shard
        
        # Return None to prevent gradient from propagating further
        # (since we manually handled it)
        return None
            
    def _wait_handles(self):
        """Wait for all async communication handles."""
        for h in self._handles:
            if hasattr(h, 'wait'):
                h.wait()
        self._handles.clear()
        
    def forward(self, *inputs, **kwargs):
        """Forward pass - hooks handle all-gather automatically."""
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous gradient communication to finish.
        This should be called after backward() and before optimizer.step().
        """
        self._wait_handles()
        
        # All-reduce gradients for replicated (non-FSDP) parameters
        for param in self._replicated_params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size
        
        # Ensure any remaining gathered params are freed
        for pid, info in self._param_info.items():
            if info['gathered'] is not None:
                setattr(info['module'], info['param_name'], info['sharded'])
                info['gathered'] = None