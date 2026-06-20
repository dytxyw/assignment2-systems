"""
Benchmark: Minimal DDP Flat vs Individual All-Reduce

Compares two implementations:
1. NaiveDDP: individual all-reduce per parameter
2. MinimalFlatDDP: single batched all-reduce on flattened gradients

Usage:
    python benchmark_flat_vs_individual.py --world_size 2 --model_size small --backend gloo
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

# from tests.adapters import get_ddp, ddp_on_after_backward
# from cs336_systems.ddp.minimal_flat_ddp import MinimalFlatDDP
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from tests.adapters import get_ddp, ddp_on_after_backward
from cs336_systems.ddp.minimal_flat_ddp import MinimalFlatDDP

MODEL_CONFIGS = {
    "tiny": {
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
}


class TransformerBlock(nn.Module):
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

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        b, s = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(s, device=input_ids.device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)


class BenchmarkTimer:
    def __init__(self, device):
        self.device = device
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_worker(
    rank, world_size, backend, model_size, num_steps, warmup_steps, 
    results_queue, use_flat_ddp
):
    """Benchmark either flat or individual DDP."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # Different port to avoid conflict

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if (backend == "nccl" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    is_main = rank == 0
    config = MODEL_CONFIGS[model_size]
    batch_size = config["batch_size"] if device.type == "cuda" else max(1, config["batch_size"] // 4)
    seq_len = config["seq_len"]
    vocab_size = 50257

    model = ToyLanguageModel(
        d_model=config["d_model"], num_heads=config["num_heads"],
        num_layers=config["num_layers"], d_ff=config["d_ff"], max_seq_len=seq_len
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Wrap with appropriate DDP
    if use_flat_ddp:
        ddp_model = MinimalFlatDDP(model)
    else:
        ddp_model = get_ddp(model)

    dist.barrier()

    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    if is_main:
        mode = "FLAT" if use_flat_ddp else "INDIVIDUAL"
        print(f"\n[Rank {rank}] {mode} DDP - Warming up...")

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        if use_flat_ddp:
            ddp_model.finish_gradient_synchronization()
        else:
            ddp_on_after_backward(ddp_model, optimizer)
        optimizer.step()

    dist.barrier()

    # Benchmark
    step_times = []
    comm_times = []
    compute_times = []

    for step in range(num_steps):
        with BenchmarkTimer(device) as total_t:
            with BenchmarkTimer(device) as compute_t:
                optimizer.zero_grad()
                logits = ddp_model(input_ids)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                loss.backward()

            with BenchmarkTimer(device) as comm_t:
                if use_flat_ddp:
                    ddp_model.finish_gradient_synchronization()
                else:
                    ddp_on_after_backward(ddp_model, optimizer)

            optimizer.step()

        step_times.append(total_t.elapsed_ms)
        compute_times.append(compute_t.elapsed_ms)
        comm_times.append(comm_t.elapsed_ms)

    dist.barrier()

    if is_main:
        import numpy as np
        results = {
            "mode": "flat" if use_flat_ddp else "individual",
            "model_size": model_size,
            "world_size": world_size,
            "backend": backend,
            "device": str(device),
            "num_parameters": count_parameters(model),
            "num_steps": num_steps,
            "step_time_ms": {"mean": float(np.mean(step_times)), "std": float(np.std(step_times))},
            "compute_time_ms": {"mean": float(np.mean(compute_times)), "std": float(np.std(compute_times))},
            "comm_time_ms": {"mean": float(np.mean(comm_times)), "std": float(np.std(comm_times))},
            "comm_proportion_pct": float(np.mean(comm_times) / np.mean(step_times) * 100),
        }
        results_queue.put(results)

    dist.destroy_process_group()


def run_single_benchmark(world_size, backend, model_size, num_steps, warmup_steps, use_flat):
    """Run one benchmark configuration."""
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    mp.spawn(
        benchmark_worker,
        args=(world_size, backend, model_size, num_steps, warmup_steps, results_queue, use_flat),
        nprocs=world_size,
        join=True,
    )
    return results_queue.get()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flat vs Individual DDP")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--model_size", type=str, default="tiny", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "nccl", "gloo"])
    args = parser.parse_args()

    if args.backend == "auto":
        backend = "nccl" if (torch.cuda.is_available() and torch.cuda.device_count() >= args.world_size) else "gloo"
    else:
        backend = args.backend

    print(f"\n{'='*70}")
    print(f"Benchmark: Flat vs Individual All-Reduce")
    print(f"{'='*70}")
    print(f"Model: {args.model_size}, World size: {args.world_size}, Backend: {backend}")
    print(f"Steps: {args.num_steps}, Warmup: {args.warmup_steps}")
    print(f"{'='*70}\n")

    # Run individual DDP benchmark
    print("\n>>> Running INDIVIDUAL all-reduce DDP...")
    individual_results = run_single_benchmark(
        args.world_size, backend, args.model_size, args.num_steps, args.warmup_steps, use_flat=False
    )

    # Run flat DDP benchmark
    print("\n>>> Running FLAT all-reduce DDP...")
    flat_results = run_single_benchmark(
        args.world_size, backend, args.model_size, args.num_steps, args.warmup_steps, use_flat=True
    )

    # Print comparison
    print(f"\n{'='*70}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\nParameters: {individual_results['num_parameters']:,}")
    print(f"\n{'Method':<20} {'Step Time (ms)':<18} {'Comm Time (ms)':<18} {'Comm %':<10}")
    print(f"{'-'*70}")

    for r in [individual_results, flat_results]:
        mode = r['mode'].upper()
        step = r['step_time_ms']['mean']
        comm = r['comm_time_ms']['mean']
        pct = r['comm_proportion_pct']
        print(f"{mode:<20} {step:<18.3f} {comm:<18.3f} {pct:<10.2f}")

    # Calculate speedup
    speedup = individual_results['comm_time_ms']['mean'] / flat_results['comm_time_ms']['mean']
    reduction = (1 - flat_results['comm_time_ms']['mean'] / individual_results['comm_time_ms']['mean']) * 100

    print(f"\n--- Communication Time Comparison ---")
    print(f"Flat all-reduce is {speedup:.2f}x faster in communication")
    print(f"Communication time reduced by {reduction:.1f}%")

    print(f"\n--- 1-2 Sentence Comparison ---")
    print(f"Batched flat all-reduce reduces communication time by {reduction:.1f}% compared to")
    print(f"individual per-parameter all-reduces, because it eliminates per-parameter kernel")
    print(f"launch overhead and achieves better network bandwidth utilization with larger buffers.")

    print(f"{'='*70}\n")

    # Save results
    with open("benchmark_flat_vs_individual.json", "w") as f:
        json.dump({"individual": individual_results, "flat": flat_results}, f, indent=2)
    print("Results saved to benchmark_flat_vs_individual.json")


if __name__ == "__main__":
    main()