#!/usr/bin/env python3
# resnet_inference.py
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 license.

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sys
sys.path.append("../cuda-probe-test/resnet")
from utils.readData import read_dataset
from utils.ResNet import ResNet152
import time

class HybridResNetInference:
    def __init__(self, split_layer=0, gpu_device=0, torch_dtype=torch.float16):
        self.device = f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu'
        self.split_layer = split_layer
        self.dtype = torch_dtype
        
        # Initialize model
        self.model = ResNet152()
        
        # Split model between CPU and GPU
        self._split_model()
        
        print(f"Process PID: {os.getpid()}")
        print(f"Layers 0-{split_layer-1}: CPU")
        print(f"Layers {split_layer}-end: GPU:{gpu_device}")
        print("Model loaded ✔︎\n")

    def _split_model(self):
        """Split the model between CPU and GPU."""
        # Move the entire model to CPU first
        self.model = self.model.to('cpu')
        
        # Split the model at the specified layer
        if self.split_layer > 0:
            # Move layers after split_layer to GPU
            for i in range(self.split_layer, len(self.model.layer1)):
                self.model.layer1[i] = self.model.layer1[i].to(self.device)
            for i in range(len(self.model.layer2)):
                self.model.layer2[i] = self.model.layer2[i].to(self.device)
            for i in range(len(self.model.layer3)):
                self.model.layer3[i] = self.model.layer3[i].to(self.device)
            for i in range(len(self.model.layer4)):
                self.model.layer4[i] = self.model.layer4[i].to(self.device)
            
            # Move final layers to GPU
            self.model.avgpool = self.model.avgpool.to(self.device)
            self.model.fc = self.model.fc.to(self.device)
        else:
            # If split_layer is 0, move entire model to GPU
            self.model = self.model.to(self.device)

    def _forward_pass(self, x):
        """Execute forward pass with CPU-GPU split."""
        # Initial layers on CPU
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        # Process each layer group
        for i, layer in enumerate(self.model.layer1):
            if i < self.split_layer:
                x = layer(x)
            else:
                x = x.to(self.device)
                x = layer(x)
        
        # Remaining layers on GPU
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x

    def run_inference(self, batch_size=32, num_iterations=64):
        """Run inference with the hybrid model."""
        iter_durations = []
        
        # Load dataset
        train_loader, _, _ = read_dataset(batch_size=batch_size, pic_path='dataset')
        
        self.model.eval()
        nb_iteration = 0
        
        with torch.no_grad():
            for data, target in tqdm(train_loader, total=len(train_loader)):
                start_t = time.time()
                
                # Move input data to CPU
                data = data.to('cpu')
                
                # Forward pass
                output = self._forward_pass(data)
                
                # Force synchronization
                torch.cuda.synchronize()
                
                end_t = time.time()
                iter_durations.append(int(round((end_t-start_t) * 1000)))
                
                nb_iteration += 1
                if nb_iteration >= num_iterations:
                    break
        
        # Calculate statistics
        np_iter_durations = np.array(iter_durations)
        print(
            f"Latency statistics:"
            f" p10({np.percentile(np_iter_durations, 10)} ms), "
            f" p50({np.percentile(np_iter_durations, 50)} ms), "
            f" p99({np.percentile(np_iter_durations, 99)} ms), "
            f" mean({np.mean(np_iter_durations)} ms)"
        )
        
        return np_iter_durations

def main():
    # Example usage
    split_layer = 5  # Adjust this value to change the CPU/GPU split point
    model = HybridResNetInference(split_layer=split_layer)
    model.run_inference()

if __name__ == '__main__':
    main()
