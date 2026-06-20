"""
Naïve DDP Benchmarking Script for CS336 Assignment 2

Benchmarks a language model with naïve DDP implementation.
Measures total time per training step and communication overhead.

Supports:
- Single-node multi-GPU (nccl backend)
- Single-node multi-CPU (gloo backend) -- fallback for machines without multiple GPUs
- Automatic backend selection and model size adjustment based on hardware availability

Usage:
    # 2 GPUs (if available)
    python benchmark_ddp.py --world_size 2 --model_size small

    # 2 CPU processes (fallback) - auto switches to tiny model
    python benchmark_ddp.py --world_size 2 --model_size small --backend gloo

    # Force tiny model for CPU testing
    python benchmark_ddp.py --world_size 2 --model_size tiny --backend gloo

    # Profile with detailed per-layer communication breakdown
    python benchmark_ddp.py --world_size 2 --model_size small --profile
"""

import os
import sys
import time
import argparse
import json
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

# Import your DDP adapters
from tests.adapters import get_ddp, ddp_on_after_backward


# =============================================================================
# Model Configuration (from Section 2.1.2 of the assignment)
# =============================================================================

MODEL_CONFIGS = {
    "tiny": {  # Ultra-small model for CPU testing / local validation
        "d_model": 128,
        "d_ff": 512,
        "num_layers": 2,
        "num_heads": 2,
        "seq_len": 64,
        "batch_size": 2,
    },
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "seq_len": 512,
        "batch_size": 16,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "seq_len": 512,
        "batch_size": 8,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "seq_len": 512,
        "batch_size": 4,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "seq_len": 512,
        "batch_size": 2,
    },
}


# =============================================================================
# Simple Transformer Language Model (for benchmarking)
# =============================================================================

class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0])
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class ToyLanguageModel(nn.Module):
    """Simple Transformer language model for benchmarking."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


# =============================================================================
# Benchmarking Utilities
# =============================================================================

class BenchmarkTimer:
    """Context manager for timing operations with CUDA synchronization."""

    def __init__(self, device: torch.device, name: str = ""):
        self.device = device
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_gradient_size(model: nn.Module) -> float:
    total_params = count_parameters(model)
    return total_params * 4 / (1024 ** 2)


def get_memory_info():
    """Get current system memory info."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": mem.percent,
        }
    else:
        return {"total_gb": 0, "available_gb": 0, "percent_used": 0}


# =============================================================================
# Core Benchmarking Function (runs in each process)
# =============================================================================

def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    model_size: str,
    num_steps: int,
    warmup_steps: int,
    results_queue: mp.Queue,
    profile: bool = False,
):
    """Worker function that runs in each distributed process."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if backend == "nccl" and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    is_main = rank == 0

    # -------------------------------------------------------------------------
    # Create model and data
    # -------------------------------------------------------------------------
    config = MODEL_CONFIGS[model_size]
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    vocab_size = 50257

    model = ToyLanguageModel(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ddp_model = get_ddp(model)

    dist.barrier()

    # Generate synthetic data
    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------
    if is_main:
        print(f"\n[Rank {rank}] Warming up for {warmup_steps} steps...")

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        ddp_on_after_backward(ddp_model, optimizer)
        optimizer.step()

    dist.barrier()
    if is_main:
        print(f"[Rank {rank}] Warmup complete. Starting benchmark...")

    # -------------------------------------------------------------------------
    # Benchmark loop
    # -------------------------------------------------------------------------
    step_times: List[float] = []
    compute_times: List[float] = []
    comm_times: List[float] = []
    optimizer_times: List[float] = []
    param_comm_times: Dict[str, List[float]] = {} if (profile and is_main) else None

    for step in range(num_steps):
        with BenchmarkTimer(device, "total_step") as total_timer:
            with BenchmarkTimer(device, "compute") as compute_timer:
                optimizer.zero_grad()
                logits = ddp_model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                loss.backward()

            with BenchmarkTimer(device, "communication") as comm_timer:
                if profile and is_main:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            t_start = time.perf_counter()
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size
                            t_end = time.perf_counter()
                            if name not in param_comm_times:
                                param_comm_times[name] = []
                            param_comm_times[name].append((t_end - t_start) * 1000)
                else:
                    ddp_on_after_backward(ddp_model, optimizer)

            with BenchmarkTimer(device, "optimizer") as opt_timer:
                optimizer.step()

        step_times.append(total_timer.elapsed_ms)
        compute_times.append(compute_timer.elapsed_ms)
        comm_times.append(comm_timer.elapsed_ms)
        optimizer_times.append(opt_timer.elapsed_ms)

    dist.barrier()

    # -------------------------------------------------------------------------
    # Only main rank collects and reports results
    # -------------------------------------------------------------------------
    if is_main:
        import numpy as np

        step_arr = np.array(step_times)
        compute_arr = np.array(compute_times)
        comm_arr = np.array(comm_times)
        opt_arr = np.array(optimizer_times)

        results = {
            "model_size": model_size,
            "world_size": world_size,
            "backend": backend,
            "device": str(device),
            "num_parameters": count_parameters(model),
            "gradient_size_mb": estimate_gradient_size(model),
            "batch_size_per_rank": batch_size,
            "total_batch_size": batch_size * world_size,
            "seq_len": seq_len,
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "step_time_ms": {
                "mean": float(step_arr.mean()),
                "std": float(step_arr.std()),
                "min": float(step_arr.min()),
                "max": float(step_arr.max()),
            },
            "compute_time_ms": {
                "mean": float(compute_arr.mean()),
                "std": float(compute_arr.std()),
            },
            "comm_time_ms": {
                "mean": float(comm_arr.mean()),
                "std": float(comm_arr.std()),
            },
            "optimizer_time_ms": {
                "mean": float(opt_arr.mean()),
                "std": float(opt_arr.std()),
            },
            "comm_proportion_pct": float(comm_arr.mean() / step_arr.mean() * 100),
            "throughput_tokens_per_sec": float(
                batch_size * world_size * seq_len / (step_arr.mean() / 1000)
            ),
            "raw_step_times": step_arr.tolist(),
            "raw_comm_times": comm_arr.tolist(),
        }

        if param_comm_times:
            results["per_parameter_comm_ms"] = {
                name: {"mean": sum(times)/len(times), "max": max(times)}
                for name, times in param_comm_times.items()
            }

        results_queue.put(results)

    dist.destroy_process_group()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Naïve DDP Benchmarking")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes (GPUs/CPUs)")
    parser.add_argument("--model_size", type=str, default="small", choices=list(MODEL_CONFIGS.keys()),
                        help="Model size. Use 'tiny' for CPU testing to avoid OOM")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps before measurement")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "nccl", "gloo"],
                        help="Distributed backend (auto selects nccl if GPUs available)")
    parser.add_argument("--profile", action="store_true", help="Enable per-parameter communication profiling")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Auto-select backend
    if args.backend == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= args.world_size:
            backend = "nccl"
        else:
            backend = "gloo"
            if torch.cuda.is_available():
                print(f"WARNING: Only {torch.cuda.device_count()} GPU(s) available, "
                      f"but world_size={args.world_size}. Falling back to CPU (gloo).")
    else:
        backend = args.backend

    # Check memory and warn about CPU OOM
    mem_info = get_memory_info()
    if backend == "gloo":
        if args.model_size != "tiny":
            print(f"\nWARNING: Running on CPU with '{args.model_size}' model.")
            print(f"         Available RAM: {mem_info['available_gb']:.1f} GB")
            print(f"         This may cause OOM. Consider using --model_size tiny")
            print(f"         (tiny: d_model=128, num_layers=2, ~0.5M params)")
            print(f"         Press Ctrl+C within 3 seconds to cancel...")
            try:
                time.sleep(3)
            except KeyboardInterrupt:
                print("\nCancelled.")
                sys.exit(0)

    print(f"\n{'='*60}")
    print(f"Naïve DDP Benchmark")
    print(f"{'='*60}")
    print(f"Model size:      {args.model_size}")
    print(f"World size:      {args.world_size}")
    print(f"Backend:         {backend}")
    print(f"Device:          {'GPU (CUDA)' if backend == 'nccl' else 'CPU'}")
    print(f"Benchmark steps: {args.num_steps}")
    print(f"Warmup steps:    {args.warmup_steps}")
    print(f"{'='*60}\n")

    # Run benchmark
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    mp.spawn(
        benchmark_worker,
        args=(args.world_size, backend, args.model_size, args.num_steps, args.warmup_steps, results_queue, args.profile),
        nprocs=args.world_size,
        join=True,
    )

    results = results_queue.get()

    # Print results
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Model:           {args.model_size}")
    print(f"Parameters:      {results['num_parameters']:,}")
    print(f"Gradient size:   {results['gradient_size_mb']:.2f} MB")
    print(f"Batch per rank:  {results['batch_size_per_rank']}")
    print(f"Total batch:     {results['total_batch_size']}")
    print(f"\n--- Timing (mean ± std across {args.num_steps} steps) ---")
    print(f"Step time:       {results['step_time_ms']['mean']:.3f} ± {results['step_time_ms']['std']:.3f} ms")
    print(f"  Compute (F+B): {results['compute_time_ms']['mean']:.3f} ± {results['compute_time_ms']['std']:.3f} ms")
    print(f"  Communication: {results['comm_time_ms']['mean']:.3f} ± {results['comm_time_ms']['std']:.3f} ms")
    print(f"  Optimizer:     {results['optimizer_time_ms']['mean']:.3f} ± {results['optimizer_time_ms']['std']:.3f} ms")
    print(f"\n--- Overhead Analysis ---")
    print(f"Comm proportion: {results['comm_proportion_pct']:.2f}%")
    print(f"Throughput:      {results['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"{'='*60}\n")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output}")

    if "per_parameter_comm_ms" in results:
        print(f"\n--- Per-Parameter Communication Time (top 10) ---")
        sorted_params = sorted(
            results["per_parameter_comm_ms"].items(),
            key=lambda x: x[1]["mean"],
            reverse=True,
        )[:10]
        for name, stats in sorted_params:
            print(f"  {name:50s} {stats['mean']:.3f} ms (max: {stats['max']:.3f} ms)")


if __name__ == "__main__":
    main()