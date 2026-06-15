import time
import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.model import scaled_dot_product_attention
import argparse
from tqdm import tqdm

batch_size = 8          # 固定批次大小
num_heads = 1           # 单头（但代码中未实际使用）
d_models = [16, 32, 64, 128]      # 嵌入维度
seq_lens = [256, 1024, 4096, 8192] # 序列长度（缺少16384）
loop = 100              # 迭代次数
warm_up = 5    

