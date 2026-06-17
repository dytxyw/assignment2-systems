"""
PyTorch Attention Benchmarking Script
- Batch size fixed to 8, no multihead attention (single head)
- Cartesian product of d_model and seq_len configurations
- 100 forward passes + 100 backward passes
- Memory measurement before backward pass starts
- Warmup and torch.cuda.synchronize() after each pass
- Handles OOM errors gracefully
"""

import torch
import timeit
import pandas as pd
from itertools import product


# ============ Configuration ============
BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_RUNS = 100
WARMUP_RUNS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 确保使用 GPU
assert DEVICE == 'cuda', 'No GPU available! This benchmark requires CUDA.'


# ============ Attention Implementation ============
# 使用题目要求的标准 PyTorch attention（或替换为你的自定义实现）
def pytorch_attention(q, k, v, is_causal=False):
    """
    Standard scaled dot-product attention (single head, no multihead)
    q, k, v shape: (batch_size, seq_len, d_model)
    """
    d_model = q.shape[-1]
    scale = 1.0 / (d_model ** 0.5)
    
    # Compute attention scores: (batch, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Optional causal mask (not required by problem, but included for completeness)
    if is_causal:
        seq_len = q.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), 
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax and output
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


# ============ Benchmarking Function ============
def benchmark_attention(d_model, seq_len):
    """
    Benchmark attention for a specific configuration.
    Returns dict with timing and memory results, or OOM status.
    """
    print(f"\nBenchmarking: d_model={d_model}, seq_len={seq_len}")
    
    # Result template
    result = {
        'd_model': d_model,
        'seq_len': seq_len,
        'batch_size': BATCH_SIZE,
        'forward_ms_mean': None,
        'forward_ms_std': None,
        'backward_ms_mean': None,
        'backward_ms_std': None,
        'fwd_peak_mem_MB': None,      # Peak memory after forward (before backward)
        'bwd_peak_mem_MB': None,      # Peak memory after backward
        'mem_before_backward_MB': None, # Memory right before backward starts
        'status': 'SUCCESS'
    }
    
    try:
        # Create inputs: (batch_size, seq_len, d_model)
        # requires_grad=True for backward pass
        Q = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        K = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        V = torch.randn(
            BATCH_SIZE, seq_len, d_model,
            dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        
        # ---- Warmup ----
        print(f"  Warmup ({WARMUP_RUNS} runs)...")
        for _ in range(WARMUP_RUNS):
            out = pytorch_attention(Q, K, V)
            loss = out.sum()
            loss.backward()
            # Clear gradients for next iteration
            Q.grad = None
            K.grad = None
            V.grad = None
        
        # ---- Forward Pass Benchmarking ----
        print(f"  Forward pass ({NUM_RUNS} runs)...")
        forward_times = []
        
        # Reset memory stats before forward benchmarking
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
        
        for i in range(NUM_RUNS):
            # Clear gradients
            Q.grad = K.grad = V.grad = None
            
            # Synchronize before timing
            torch.cuda.synchronize(DEVICE)
            
            start = timeit.default_timer()
            out = pytorch_attention(Q, K, V)
            torch.cuda.synchronize(DEVICE)
            end = timeit.default_timer()
            
            forward_times.append((end - start) * 1000)  # Convert to ms
            
            # Keep the computation graph for backward (don't delete out)
            # But we need to clean up periodically to avoid OOM during forward loop
            if i < NUM_RUNS - 1:  # Not the last iteration
                # We need to do backward to free the graph, or detach
                # Actually, for pure forward timing, we should not accumulate graphs
                # Let's use a simpler approach: backward and clear immediately
                loss = out.sum()
                loss.backward()
                Q.grad = K.grad = V.grad = None
        
        # Record peak memory after forward phase (before backward)
        # Note: The last forward created 'out' which is still in memory
        fwd_peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        result['fwd_peak_mem_MB'] = round(fwd_peak_mem, 2)
        
        # ---- Memory Measurement Before Backward ----
        # Create a fresh forward pass to measure memory state right before backward
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
        
        # Do one forward pass and hold the output
        out = pytorch_attention(Q, K, V)
        torch.cuda.synchronize(DEVICE)
        
        # Record memory right before backward starts
        mem_before_bwd = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
        result['mem_before_backward_MB'] = round(mem_before_bwd, 2)
        
        # ---- Backward Pass Benchmarking ----
        print(f"  Backward pass ({NUM_RUNS} runs)...")
        backward_times = []
        
        # Reset peak memory stats for backward phase
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
        for i in range(NUM_RUNS):
            # Recreate forward output (backward needs fresh graph)
            # Or reuse if we can, but gradients accumulate
            # Safer: do forward + backward each time, but only time backward
            
            # Option: Time only backward, but we need a fresh out each time
            # Because after backward, the computation graph is freed
            out = pytorch_attention(Q, K, V)
            loss = out.sum()
            
            # Synchronize before timing backward
            torch.cuda.synchronize(DEVICE)
            
            back_start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize(DEVICE)
            back_end = timeit.default_timer()
            
            backward_times.append((back_end - back_start) * 1000)  # Convert to ms
            
            # Clear gradients for next iteration
            Q.grad = K.grad = V.grad = None
        
        # Record peak memory after backward phase
        bwd_peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        result['bwd_peak_mem_MB'] = round(bwd_peak_mem, 2)
        
        # Calculate statistics
        import statistics
        result['forward_ms_mean'] = round(statistics.mean(forward_times), 3)
        result['forward_ms_std'] = round(statistics.stdev(forward_times), 3) if len(forward_times) > 1 else 0.0
        result['backward_ms_mean'] = round(statistics.mean(backward_times), 3)
        result['backward_ms_std'] = round(statistics.stdev(backward_times), 3) if len(backward_times) > 1 else 0.0
        
        print(f"  ✓ Forward: {result['forward_ms_mean']:.3f} ± {result['forward_ms_std']:.3f} ms")
        print(f"  ✓ Backward: {result['backward_ms_mean']:.3f} ± {result['backward_ms_std']:.3f} ms")
        print(f"  ✓ Mem before backward: {result['mem_before_backward_MB']:.1f} MB")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ OOM: {e}")
        result['status'] = 'OOM'
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  ✗ OOM: {e}")
            result['status'] = 'OOM'
            torch.cuda.empty_cache()
        else:
            print(f"  ✗ Error: {e}")
            result['status'] = f'ERROR: {e}'
            raise
    
    return result


# ============ Main Execution ============
def main():
    print("=" * 80)
    print("PyTorch Attention Benchmarking")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"d_models: {D_MODELS}")
    print(f"seq_lengths: {SEQ_LENS}")
    print(f"Runs per config: {NUM_RUNS}")
    print("=" * 80)
    
    # Run benchmarks for all configurations
    results = []
    total_configs = len(D_MODELS) * len(SEQ_LENS)
    
    for idx, (d_model, seq_len) in enumerate(product(D_MODELS, SEQ_LENS), 1):
        print(f"\n[{idx}/{total_configs}] ", end="")
        result = benchmark_attention(d_model, seq_len)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    column_order = [
        'd_model', 'seq_len', 'batch_size', 'status',
        'forward_ms_mean', 'forward_ms_std',
        'backward_ms_mean', 'backward_ms_std',
        'fwd_peak_mem_MB', 'mem_before_backward_MB', 'bwd_peak_mem_MB'
    ]
    df = df[column_order]
    
    # Save to markdown (for report)
    with open('attention_benchmark_results.md', 'w') as f:
        f.write("# PyTorch Attention Benchmarking Results\n\n")
        f.write(f"**Device:** {torch.cuda.get_device_name(0)}\n\n")
        f.write(f"**Batch Size:** {BATCH_SIZE}\n\n")
        f.write(f"**Number of Runs:** {NUM_RUNS}\n\n")
        f.write("## Results Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # Add OOM summary
        oom_count = (df['status'] == 'OOM').sum()
        f.write(f"## Summary\n\n")
        f.write(f"- Total configurations: {total_configs}\n")
        f.write(f"- Successful: {total_configs - oom_count}\n")
        f.write(f"- OOM: {oom_count}\n")
        
        if oom_count > 0:
            f.write("\n### OOM Configurations\n\n")
            oom_df = df[df['status'] == 'OOM'][['d_model', 'seq_len']]
            f.write(oom_df.to_markdown(index=False))
    
    # Also save to CSV for data analysis
    df.to_csv('attention_benchmark_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("Benchmarking Complete!")
    print("Results saved to:")
    print("  - attention_benchmark_results.md (report)")
    print("  - attention_benchmark_results.csv (data)")
    print("=" * 80)
    
    # Print final table
    print("\nFinal Results:")
    print(df.to_string())
    
    return df


if __name__ == '__main__':
    main()