# CUDA Neuro-Surgeon

A PyTorch implementation of a hybrid CPU-GPU neural network model with dynamic, energy-aware computation scheduling.

## Overview

CUDA Neuro-Surgeon demonstrates a flexible architecture for neural networks where different components can dynamically execute on either CPU or GPU based on real-time energy and resource considerations. This approach enables:

- Energy-efficient ML inference and training
- Better resource utilization across heterogeneous hardware
- Dynamic offloading between devices at runtime

The implementation uses PyTorch 1.3+ and provides a seamless interface for end-to-end training across CPU and GPU components with a unified gradient flow.

## Features

- **Hybrid Execution**: Run AMX blocks on CPU and MoE (Mixture of Experts) blocks on GPU with seamless tensor transfer
- **Energy-Aware Scheduling**: Dynamically decide which hardware should execute each component based on power draw and utilization
- **Single Weight Source**: Maintain a canonical set of weights on CPU with GPU mirrors
- **Cross-Device Autograd**: Properly handle backpropagation across CPU and GPU computation boundaries
- **Non-Blocking Transfers**: Optimize data movement with asynchronous CPU-GPU transfers

## Requirements

- Python 3.6+
- PyTorch 1.3+
- CUDA-compatible GPU
- psutil (for CPU monitoring)
- NVIDIA management library (for GPU power monitoring)

```bash
pip install torch psutil
```

## Usage

### Basic Initialization

```python
from inference import AMXModule, MoEModule, HybridModel

# Create CPU components
amx_cpu = AMXModule(in_dim=128, hidden_dim=256).cpu()
moe_cpu = MoEModule(hidden_dim=256, out_dim=64, num_experts=8).cpu()

# Create GPU mirror of MoE component
moe_gpu = copy.deepcopy(moe_cpu).to('cuda:0')

# Build hybrid model
model = HybridModel(amx_cpu, moe_cpu, moe_gpu)
```

### Training Example

```python
from inference import train_hybrid_model
import torch.utils.data

# Create your dataset and dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, pin_memory=True
)

# Train with automatic device placement
train_hybrid_model(model, dataloader, num_epochs=5, lr=1e-3)
```

## Implementation Details

### Architecture

The implementation consists of three main components:

1. **AMXModule**: A neural network block that primarily runs on CPU
2. **MoEModule**: A Mixture of Experts block with dynamic expert routing that primarily runs on GPU
3. **HybridModel**: A wrapper that orchestrates tensor movement and execution between devices

### Energy-Aware Scheduling

The system makes per-batch decisions on where to run each component using:

- `should_offload_amx_to_cpu()`: Determines if AMX block should run on CPU
- `should_offload_moe_to_cpu()`: Determines if MoE block should run on CPU instead of GPU

These decisions are based on real-time monitoring of:
- GPU power draw via `nvidia-smi`
- CPU utilization percentage via `psutil`

### Cross-Device Training

The training loop handles:
1. Forward pass across CPU and GPU components
2. Backward pass with automatic gradient propagation across devices
3. Parameter updates on the CPU master copy
4. Synchronization of updated weights to GPU mirrors

## Performance Considerations

- Use `non_blocking=True` for all device transfers to overlap computation and communication
- Enable `pin_memory=True` in your DataLoader for faster CPU→GPU transfers
- Consider batch size carefully as it affects both computation efficiency and transfer overhead
- For large models, you may synchronize weights less frequently if slight staleness can be tolerated

## Customization

You can modify the energy thresholds in the scheduling functions:

```python
def should_offload_moe_to_cpu(gpu_power_threshold=200.0, cpu_util_threshold=80.0):
    # Lower gpu_power_threshold to be more aggressive about saving GPU energy
    # Adjust cpu_util_threshold based on your CPU capabilities
```

## Extensions

- Add more components with their own placement strategies
- Implement finer-grained partitioning within modules
- Extend to multi-GPU scenarios with distributed training
- Explore quantization for CPU components to improve efficiency 