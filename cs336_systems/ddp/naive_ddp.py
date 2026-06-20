import torch
import torch.distributed as dist  # 分布式通信的核心
from typing import List


class naiveddp(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized")

        self.module = module
        self.world_size = dist.get_world_size()  # 获取分布式环境中的总进程数（即 GPU 总数）

        # Broadcast initial parameters from rank 0
        self._broadcast_from_rank0()

    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            if b is not None and torch.is_tensor(b):
                dist.broadcast(b, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        if self.world_size <= 1:
            return
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size