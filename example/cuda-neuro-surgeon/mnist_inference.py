#!/usr/bin/env python3
# mnist_hybrid_inference.py
#
# Hybrid CPU↔GPU inference for a 2-layer MLP on MNIST.
# Layers before `split_layer` stay on CPU, the rest move to GPU 0.
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 License.

import os, time, gc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ── configuration ──────────────────────────────────────────────────────────
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE          = 128
MNIST_DATA_PATH     = "../mix-schedule-mnist/train/data"
MODEL_LOAD_PATH     = "../mix-schedule-mnist/train/mnist_mlp_pytorch.pth"
MAX_IMAGES_TO_TEST  = 2_000
GPU_ID              = 0
DTYPE               = torch.float32          # FP32 suits MNIST

# ── model definition ───────────────────────────────────────────────────────
class HybridMLP(nn.Module):
    """
    784 → 128 → 10 MLP with optional CPU/GPU split.
    split_layer:
        0 → fc1+relu+fc2 on GPU
        1 → fc1 on CPU,   relu+fc2 on GPU
        2 → fc1+relu on CPU, fc2 on GPU
    """
    def __init__(self, split_layer: int = 0,
                 gpu_device: int = 0,
                 torch_dtype = torch.float32):
        super().__init__()
        self.split_layer = split_layer
        self.gpu         = (torch.device(f"cuda:{gpu_device}")
                            if torch.cuda.is_available() else torch.device("cpu"))

        self.fc1  = nn.Linear(784, 128, dtype=torch_dtype)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(128, 10,  dtype=torch_dtype)

        self._place_layers()
        print(f"Model split at layer {split_layer}")
        print(f"  Layers 0–{max(split_layer-1,0)} : CPU")
        print(f"  Layers {split_layer}–end : GPU:{gpu_device}\n")

    def _place_layers(self):
        # move everything to CPU first
        for layer in (self.fc1, self.relu, self.fc2):
            layer.to("cpu")

        # then move GPU part
        if self.split_layer == 0:
            for layer in (self.fc1, self.relu, self.fc2):
                layer.to(self.gpu)
        elif self.split_layer == 1:
            self.relu.to(self.gpu)
            self.fc2.to(self.gpu)
        elif self.split_layer == 2:
            self.fc2.to(self.gpu)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure tensor is where fc1 lives
        x = x.to(self.fc1.weight.device, non_blocking=True)

        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)

        # ensure tensor is where fc2 lives
        x = x.to(self.fc2.weight.device, non_blocking=True)
        x = self.fc2(x)
        return x

# ── inference routine ─────────────────────────────────────────────────────
@torch.inference_mode()
def run_inference(model: nn.Module, test_loader: DataLoader):
    correct, total = 0, 0
    latencies_ms   = []

    model.eval()
    on_cuda = any(p.is_cuda for p in model.parameters())

    for images, labels in test_loader:
        start = time.perf_counter()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        if on_cuda:
            torch.cuda.synchronize()

        latencies_ms.append((time.perf_counter() - start) * 1e3)
        total   += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()

    acc  = 100 * correct / total
    avg  = sum(latencies_ms) / len(latencies_ms)
    return acc, avg, total

# ── main driver ───────────────────────────────────────────────────────────
def main():
    print(f"Running on: {DEVICE}\n")

    # data -----------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_test = torchvision.datasets.MNIST(
        root      = MNIST_DATA_PATH,
        train     = False,
        download  = True,
        transform = transform
    )

    n_test = min(len(full_test), MAX_IMAGES_TO_TEST)
    test_loader = DataLoader(
        Subset(full_test, range(n_test)),
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers= 2,
        pin_memory = torch.cuda.is_available()
    )

    # evaluate several split points ---------------------------------------
    results = []
    for split in (0, 1, 2):
        print(f"\n── Split point {split} ───────────────────────────────")
        model = HybridMLP(split_layer=split,
                          gpu_device = GPU_ID,
                          torch_dtype=DTYPE)

        if not os.path.exists(MODEL_LOAD_PATH):
            print(f"❌  model file not found: {MODEL_LOAD_PATH}")
            return

        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location="cpu"))

        acc, lat, tot = run_inference(model, test_loader)
        results.append((split, acc, lat))

        print(f"Accuracy         : {acc:.2f}%")
        print(f"Avg latency / b. : {lat:.2f} ms over {tot} images")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # summary -------------------------------------------------------------
    print("\nSummary:")
    for s, acc, lat in results:
        print(f"  split={s} → acc={acc:.2f}%  avg-lat={lat:.2f} ms")

if __name__ == "__main__":
    main()
