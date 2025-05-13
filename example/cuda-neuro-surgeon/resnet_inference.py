#!/usr/bin/env python3
# hybrid_resnet_inference.py
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 license.

import os, time, sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

# ── local package -------------------------------------------------------------
sys.path.append("../cuda-probe-test/resnet")
from utils.ResNet import ResNet152          # your own implementation
from utils.readData import read_dataset     # returns (train, val, test) loaders
# ───────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------#
#  HybridResNetInference
# -----------------------------------------------------------------------------#
class HybridResNetInference:
    """
    Run ResNet152 with layers [0 .. split_index-1] on the CPU
                       and layers [split_index .. end] on GPU <gpu_device>.
    The split happens at the granularity returned by `model.children()`,
    which for ResNet is::
        [conv1, bn1, relu, maxpool?, layer1, layer2, layer3, layer4,
         avgpool, fc]
    """
    def __init__(self, split_index: int = 4,
                 gpu_device: int = 0,
                 torch_dtype=torch.float16) -> None:

        self.cpu = torch.device('cpu')
        self.gpu = torch.device(f'cuda:{gpu_device}'
                                if torch.cuda.is_available() else 'cpu')
        self.dtype = torch_dtype
        self.split_index = split_index

        # ---- build the two sequential parts ---------------------------------
        full_model = ResNet152()                 # weights are on CPU by default
        children   = list(full_model.children())

        self.cpu_part = nn.Sequential(*children[:split_index]).to(self.cpu)
        self.gpu_part = nn.Sequential(*children[split_index:-2]).to(self.gpu,
                                                                    dtype=self.dtype)
        self.avgpool  = children[-2].to(self.gpu)
        self.fc       = children[-1].to(self.gpu, dtype=self.dtype)
        self.cpu_part.eval()
        self.gpu_part.eval()

        print(f"Process PID          : {os.getpid()}")
        print(f"Layers 0-{split_index-1:>2}   → CPU")
        print(f"Layers {split_index:>2}-end → GPU:{gpu_device}")
        print("Model ready ✔︎\n")

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cpu_part(x)                                # CPU
        x = x.to(self.gpu, dtype=self.dtype, non_blocking=True)
        x = self.gpu_part(x)                                # GPU body
        x = self.avgpool(x)
        x = torch.flatten(x, 1)                             # flatten N,2048
        x = self.fc(x)                                      # logits
        return x

    # -------------------------------------------------------------------------
    def run_inference(self,
                      batch_size: int = 32,
                      num_batches: int = 64,
                      data_root: str = "dataset") -> np.ndarray:

        train_loader, _, _ = read_dataset(batch_size=batch_size,
                                          pic_path=data_root)
        iter_ms = []

        with torch.inference_mode():
            for i, (data, _) in enumerate(tqdm(train_loader,
                                               total=min(num_batches,
                                                         len(train_loader)),
                                               unit="batch")):
                if i >= num_batches:
                    break

                start = time.perf_counter()

                logits = self.forward(data.to(self.cpu))   # CPU input

                torch.cuda.synchronize(self.gpu)           # accurate timing
                dur_ms = (time.perf_counter() - start) * 1e3
                iter_ms.append(dur_ms)

        arr = np.asarray(iter_ms)
        print(f"Latency statistics:"
              f" p10({np.percentile(arr, 10):.1f} ms),"
              f" p50({np.percentile(arr, 50):.1f} ms),"
              f" p99({np.percentile(arr, 99):.1f} ms),"
              f" mean({arr.mean():.1f} ms)")
        return arr

# -----------------------------------------------------------------------------#
#  main
# -----------------------------------------------------------------------------#
def main() -> None:
    """
    Adjust `split_index` to choose how much of the network stays on CPU.
    Typical ResNet152 `children()` layout (index : module):

      0 conv1       5 layer2
      1 bn1         6 layer3
      2 relu        7 layer4
      3 (maybe maxpool)*
      4 layer1      8 avgpool
                    9 fc

    If your `ResNet152` variant omits the top-level `maxpool`, indices shift
    down by one – inspect `list(ResNet152().children())` to be sure.
    """
    SPLIT_INDEX = 4           # everything before layer1 on CPU
    GPU_DEVICE  = 0

    engine = HybridResNetInference(split_index=SPLIT_INDEX,
                                   gpu_device=GPU_DEVICE,
                                   torch_dtype=torch.float16)

    # run a quick measurement pass
    engine.run_inference(batch_size=32, num_batches=64)

if __name__ == "__main__":
    main()
