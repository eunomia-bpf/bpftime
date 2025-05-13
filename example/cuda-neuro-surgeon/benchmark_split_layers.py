#!/usr/bin/env python3
# benchmark_split_layers.py
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 license.

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from llama_inference import HybridLLaMAInference, MODEL_PATH, TOKENIZER_PATH, EXAMPLE_PROMPT

def run_benchmark(split_layer, num_runs=3):
    """Run benchmark for a specific split layer configuration."""
    print(f"\nTesting split layer: {split_layer}")
    
    # Initialize model with current split layer
    engine = HybridLLaMAInference(
        MODEL_PATH,
        TOKENIZER_PATH,
        split_layer=split_layer,
        gpu_device=0,
        torch_dtype=torch.float16
    )
    
    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        outputs = engine.infer(EXAMPLE_PROMPT, batch_size=1, max_new_tokens=256, stream=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"Run {i+1}/{num_runs}: {elapsed:.2f}s")
    
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

def main():
    # Test different split layer configurations
    split_layers = range(5, 41, 5)  # From 5 to 40
    results = []
    
    for split_layer in split_layers:
        latency = run_benchmark(split_layer)
        results.append({
            'split_layer': split_layer,
            'latency': latency
        })
        print(f"Average latency for split layer {split_layer}: {latency:.2f}s")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = 'split_layer_performance.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate performance graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['split_layer'], df['latency'], 'b-o')
    plt.xlabel('Split Layer (CPU/GPU Boundary)')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    
    # Save the plot
    plot_path = 'split_layer_performance.pdf'
    plt.savefig(plot_path)
    print(f"Performance graph saved to {plot_path}")

if __name__ == "__main__":
    main() 