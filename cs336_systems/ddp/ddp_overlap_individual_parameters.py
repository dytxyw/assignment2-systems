"""
DDP with Overlapping Individual Parameters.

Implements a DDP wrapper that overlaps gradient communication with backward pass
computation by asynchronously all-reducing each parameter's gradient as soon as it
is ready.
"""
import torch
import torch.distributed as dist
from typing import List


class DDPOverlapIndividualParameters(torch.nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps communication with computation.
    
    For each parameter, as soon as its gradient is accumulated during the backward pass,
    an asynchronous all-reduce is launched. This allows the communication to overlap
    with the remaining backward computation.
    
    Args:
        module: The PyTorch module to wrap.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        
        if not dist.is_initialized():
            raise RuntimeError(
                "Process group not initialized; call dist.init_process_group first."
            )
        
        self.module = module
        self.world_size = dist.get_world_size()
        
        # Store async handles for all-reduce operations
        self._handles: List[dist.Work] = []
        
        # Broadcast initial parameters from rank 0 to all ranks
        self._broadcast_from_rank0()
        
        # Register hooks to trigger async all-reduce when gradients are ready
        self._register_grad_hooks()
    
    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        """Broadcast parameters and buffers from rank 0 to all other ranks."""
        # Broadcast parameters
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        
        # Broadcast buffers (for running stats in BatchNorm, etc.)
        for b in self.module.buffers():
            if b is not None and torch.is_tensor(b):
                dist.broadcast(b, src=0)
    
    def _register_grad_hooks(self) -> None:
        """Register post-accumulate-grad hooks for async gradient all-reduce."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._on_grad_ready)
    
    def _on_grad_ready(self, param: torch.nn.Parameter) -> None:
        """
        Callback triggered when a parameter's gradient is accumulated.
        
        Launches an asynchronous all-reduce on the gradient. The all-reduce
        computes the sum across ranks; we divide by world_size here so that
        the final gradient is averaged.
        
        Args:
            param: The parameter whose gradient just became ready.
        """
        # Ensure grad exists
        if param.grad is None:
            return
        
        # Scale gradient by 1/world_size so that after SUM all-reduce,
        # each rank has the average gradient.
        if self.world_size > 1:
            param.grad.div_(self.world_size)
        
        # Launch asynchronous all-reduce (SUM)
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)
    
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all asynchronous gradient all-reduce operations to complete.
        
        Must be called after backward() and before optimizer.step().
        """
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
    
    def forward(self, *inputs, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*inputs, **kwargs)