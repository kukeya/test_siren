import subprocess
import numpy as np
import time
import random
import torch

def auto_select_gpu(min_memory_mb=8000):
    """
    Auto-select GPU based on utilization and memory availability.
    Strategy:
    1. Filter GPUs with enough free memory (> min_memory_mb).
    2. From those, select the one with the lowest GPU utilization.
    3. If multiple have same lowest utilization, pick the one with most free memory.
    """
    # Random sleep to avoid race conditions when launching multiple jobs
    time.sleep(random.uniform(0, 10))
    try:
        # Query memory.free and utilization.gpu
        cmd = 'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,nounits,noheader'
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        lines = result.strip().split('\n')
        
        candidates = []
        for i, line in enumerate(lines):
            mem_free, util = map(int, line.split(','))
            if mem_free >= min_memory_mb:
                candidates.append((i, mem_free, util))
        
        if not candidates:
            print(f"No GPU found with > {min_memory_mb} MB free memory. Fallback to max memory strategy.")
            # Fallback: just pick max memory
            memory_free = [int(line.split(',')[0]) for line in lines]
            gpu_id = int(np.argmax(memory_free))
        else:
            # Sort by utilization (asc), then by memory free (desc)
            # x[2] is util, x[1] is mem_free
            candidates.sort(key=lambda x: (x[2], -x[1]))
            gpu_id = candidates[0][0]
            best_mem = candidates[0][1]
            best_util = candidates[0][2]
            print(f"Auto-selecting GPU {gpu_id}: Util {best_util}%, Free Mem {best_mem} MB")

        torch.cuda.set_device(gpu_id)
    except Exception as e:
        print(f"Auto-select GPU failed: {e}")