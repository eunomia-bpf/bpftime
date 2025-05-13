#!/usr/bin/env python3
# benchmark_resnet_splits.py
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 license.

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from resnet_inference import HybridResNetInference

def run_benchmark(split_index, num_runs=3, batch_size=32, num_batches=64):
    """Run benchmark for a specific split layer configuration."""
    print(f"\nTesting split index: {split_index}")
    
    # Initialize model with current split index
    engine = HybridResNetInference(
        split_index=split_index,
        gpu_device=0,
        torch_dtype=torch.float16
    )
    
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        # Run inference
        engine.run_inference(batch_size=batch_size, num_batches=num_batches)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"Run {i+1}/{num_runs}: {elapsed:.2f}s")
        
        torch.cuda.empty_cache()
    
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

def main():
    """
    Test different split layer configurations for ResNet152.
    Typical ResNet152 children() layout (index : module):
      0 conv1       5 layer2
      1 bn1         6 layer3
      2 relu        7 layer4
      3 maxpool     8 avgpool
      4 layer1      9 fc
    """
    # Test different split configurations
    split_indices = range(0, 10)  # Test all possible split points
    results = []
    
    for split_index in split_indices:
        latency = run_benchmark(split_index)
        results.append({
            'split_index': split_index,
            'latency': latency
        })
        print(f"Average latency for split index {split_index}: {latency:.2f}s")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = 'resnet_split_performance.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate performance graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['split_index'], df['latency'], 'b-o')
    plt.title('ResNet152 Performance vs Split Layer Configuration')
    plt.xlabel('Split Index (CPU/GPU Boundary)')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    
    # Add module names as x-tick labels
    module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 
                   'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
    plt.xticks(range(len(module_names)), module_names, rotation=45)
    
    # Save the plot
    plot_path = 'resnet_split_performance.pdf'
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(plot_path)
    print(f"Performance graph saved to {plot_path}")

if __name__ == "__main__":
    main() 