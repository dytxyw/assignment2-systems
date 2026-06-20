"""
Minimal DDP with Flattened Gradients

Instead of individually all-reducing each parameter's gradient,
this implementation:
1. Flattens all gradients into a single contiguous tensor
2. Performs one batched all-reduce on the flat tensor
3. Unflattens the result back into parameter gradients

This is more efficient because:
- Fewer communication round-trips (1 vs N parameters)
- Better utilization of network bandwidth (larger contiguous buffers)
- Reduced kernel launch overhead
"""

import torch
import torch.distributed as dist
from typing import List


class MinimalFlatDDP(torch.nn.Module):
    """
    Minimal DDP that communicates all gradients as a single flat tensor.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized")

        self.module = module
        self.world_size = dist.get_world_size()

        # Build gradient bucket: a single flat tensor holding all gradients
        self._build_grad_bucket()

        # Broadcast initial parameters from rank 0
        self._broadcast_from_rank0()

    def _build_grad_bucket(self):
        """
        Build a flat tensor that can hold all parameter gradients.
        Also create mappings from parameter -> flat tensor slice.
        """
        # Calculate total gradient size
        total_grad_size = 0
        self._param_shapes = []  # Store shapes for unflattening
        self._param_offsets = []  # Store (start, end) offsets in flat tensor

        for p in self.module.parameters():
            if p.requires_grad:
                numel = p.numel()
                start = total_grad_size
                end = total_grad_size + numel
                self._param_shapes.append(p.shape)
                self._param_offsets.append((start, end))
                total_grad_size += numel

        self._total_grad_size = total_grad_size

        # Create the flat gradient bucket
        # Use the same dtype and device as the first parameter
        sample_param = next(self.module.parameters())
        self._grad_bucket = torch.zeros(
            total_grad_size, 
            dtype=sample_param.dtype, 
            device=sample_param.device
        )

        print(f"[MinimalFlatDDP] Built gradient bucket: {total_grad_size:,} elements "
              f"({total_grad_size * 4 / 1024**2:.2f} MB)")

    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        """Broadcast model parameters from rank 0 to all ranks."""
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            if b is not None and torch.is_tensor(b):
                dist.broadcast(b, src=0)

    def forward(self, *inputs, **kwargs):
        """Delegate forward pass to the underlying module."""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def _flatten_grads(self):
        """Copy all parameter gradients into the flat bucket."""
        for i, p in enumerate(self.module.parameters()):
            if p.grad is not None:
                start, end = self._param_offsets[i]
                self._grad_bucket[start:end].copy_(p.grad.data.view(-1))

    @torch.no_grad()
    def _unflatten_grads(self):
        """Copy flat bucket back into parameter gradients."""
        for i, p in enumerate(self.module.parameters()):
            if p.grad is not None:
                start, end = self._param_offsets[i]
                p.grad.data.copy_(self._grad_bucket[start:end].view(self._param_shapes[i]))

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """
        Synchronize gradients using a single flat all-reduce.

        Steps:
        1. Flatten all gradients into _grad_bucket
        2. all-reduce the entire bucket (SUM)
        3. Divide by world_size (average)
        4. Unflatten back into parameter gradients
        """
        if self.world_size <= 1:
            return

        # Step 1: Copy all gradients into flat bucket
        self._flatten_grads()

        # Step 2: Single all-reduce on the flat bucket
        dist.all_reduce(self._grad_bucket, op=dist.ReduceOp.SUM)

        # Step 3: Average
        self._grad_bucket /= self.world_size

        # Step 4: Copy back to parameter gradients
        self._unflatten_grads()