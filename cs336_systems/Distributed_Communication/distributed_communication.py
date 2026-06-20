import torch
import torch.distributed as dist
import time
import os
import argparse
from torch.multiprocessing import spawn
import matplotlib.pyplot as plt
import pandas as pd

def setup(master_addr, master_port, rank, world_size, backend):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_all_reduce(rank, world_size, tensor_size_mb, backend, device, master_addr, master_port, return_dict):
    """ 评测函数主体 - 使用共享字典返回结果 """
    setup(master_addr, master_port, rank, world_size, backend)
    
    if device == 'cuda':
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4
    tensor_data = torch.randn(num_elements, device=device)

    # Warm-up
    for _ in range(5):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    dist.barrier()
    
    # 正式计时
    start_time = time.time()
    num_iterations = 20
    for _ in range(num_iterations):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    duration = end_time - start_time
    avg_time = duration / num_iterations
    bandwidth_gbps = (tensor_size_bytes / avg_time) / 1e9

    if rank == 0:
        print(f"Backend: {backend}, Device: {device}, World Size: {world_size}, Tensor Size: {tensor_size_mb}MB")
        print(f"Average time per all-reduce: {avg_time * 1000:.4f} ms")
        print(f"Achieved Bandwidth: {bandwidth_gbps:.4f} GB/s\n")

    # 使用共享字典返回结果（spawn 的标准做法）
    result = {
        'rank': rank,
        'world_size': world_size,
        'backend': backend,
        'device': device,
        'tensor_size_mb': tensor_size_mb,
        'avg_time_ms': avg_time * 1000.0,
        'bandwidth_gbps': bandwidth_gbps
    }
    
    # 所有 rank 都将结果写入共享字典
    return_dict[rank] = result
    
    cleanup()

def run_experiment(world_size, tensor_size_mb, backend, device):
    """ 运行单次实验 """
    master_addr = 'localhost'
    master_port = 29500
    
    # 使用 Manager 创建共享字典
    from torch.multiprocessing import Manager
    manager = Manager()
    return_dict = manager.dict()
    
    spawn(
        benchmark_all_reduce,
        args=(world_size, tensor_size_mb, backend, device, master_addr, master_port, return_dict),
        nprocs=world_size,
        join=True
    )
    
    # 收集结果
    results = [return_dict[i] for i in range(world_size)]
    return results[0]  # 返回 rank 0 的结果（所有 rank 结果相同）

def main():
    # 实验配置
    tensor_sizes = [1, 10, 100, 1024]  # MB, 注意 1GB = 1024MB
    world_sizes = [2, 4, 6]
    backend = 'nccl'  # 或 'gloo'
    device = 'cuda'
    
    all_results = []
    
    for world_size in world_sizes:
        for tensor_size_mb in tensor_sizes:
            print(f"\nRunning: world_size={world_size}, tensor_size={tensor_size_mb}MB")
            result = run_experiment(world_size, tensor_size_mb, backend, device)
            all_results.append(result)
    
    # 创建 DataFrame
    df = pd.DataFrame(all_results)
    
    # 生成图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 延迟 vs 数据大小（不同进程数）
    for ws in world_sizes:
        data = df[df['world_size'] == ws]
        axes[0].plot(data['tensor_size_mb'], data['avg_time_ms'], 
                    marker='o', label=f'{ws} GPUs')
    axes[0].set_xlabel('Tensor Size (MB)')
    axes[0].set_ylabel('Average Time (ms)')
    axes[0].set_title('All-Reduce Latency vs Data Size')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # 图2: 带宽 vs 数据大小（不同进程数）
    for ws in world_sizes:
        data = df[df['world_size'] == ws]
        axes[1].plot(data['tensor_size_mb'], data['bandwidth_gbps'], 
                    marker='o', label=f'{ws} GPUs')
    axes[1].set_xlabel('Tensor Size (MB)')
    axes[1].set_ylabel('Bandwidth (GB/s)')
    axes[1].set_title('All-Reduce Bandwidth vs Data Size')
    axes[1].legend()
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('all_reduce_benchmark.png', dpi=150)
    print("\n图表已保存至 all_reduce_benchmark.png")
    
    # 打印表格
    print("\n实验结果汇总:")
    print(df.to_string())
    df.to_csv('all_reduce_results.csv', index=False)

if __name__ == '__main__':
    main() 