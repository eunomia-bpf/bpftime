import torch
import torch.nn as nn
import copy
import psutil
import subprocess

class AMXModule(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(AMXModule, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        # ... other layers ...

    def forward(self, x):
        return torch.relu(self.linear(x))


class MoEModule(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_experts):
        super(MoEModule, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # simple top-1 gating
        weights = torch.softmax(self.gate(x), dim=-1)
        expert_idx = torch.argmax(weights, dim=-1)
        out = torch.stack([self.experts[i](x[j]) for j, i in enumerate(expert_idx)])
        return out


class HybridModel(nn.Module):
    def __init__(self, amx_cpu, moe_cpu, moe_gpu):
        super(HybridModel, self).__init__()
        self.amx_cpu = amx_cpu
        self.moe_cpu = moe_cpu      # used only if you ever run MoE on CPU
        self.moe_gpu = moe_gpu      # the "live" GPU expert block

    def forward(self, x, use_cpu_amx=False, use_cpu_moe=False):
        # 1. AMX stage: by default runs on CPU
        if use_cpu_amx:
            out_amx = self.amx_cpu(x.to('cpu'))
        else:
            # move inputs to GPU, run a GPU copy of AMX if you had one...
            out_amx = self.amx_cpu(x.to('cpu'))  # here we only have a CPU AMX
            # you could mirror AMX to GPU similarly if needed

        # 2. MoE stage: by default on GPU
        if use_cpu_moe:
            out_moe = self.moe_cpu(out_amx)
        else:
            inp_gpu = out_amx.to('cuda:0', non_blocking=True)
            out_gpu = self.moe_gpu(inp_gpu)
            out_moe = out_gpu.to('cpu', non_blocking=True)

        return out_moe


def get_gpu_power():
    """Get GPU power draw in watts using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, 
            text=True
        )
        return float(result.stdout.strip())
    except:
        # Return a high value if unable to query (forces CPU offload)
        return 300.0


def get_cpu_utilization():
    """Get CPU utilization percentage"""
    return psutil.cpu_percent(interval=0.1)


def should_offload_amx_to_cpu(gpu_power_threshold=150.0):
    """Decide whether to run AMX on CPU based on GPU power usage"""
    # AMX runs on CPU by default in this example, but you could extend this
    return True


def should_offload_moe_to_cpu(gpu_power_threshold=200.0, cpu_util_threshold=80.0):
    """Decide whether to run MoE on CPU based on GPU power and CPU utilization"""
    gpu_power = get_gpu_power()
    cpu_util = get_cpu_utilization()
    
    # Offload to CPU if GPU is hot and CPU has capacity
    return gpu_power > gpu_power_threshold and cpu_util < cpu_util_threshold


def train_hybrid_model(model, dataloader, num_epochs=5, lr=1e-3):
    """Training loop with cross-device autograd"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # Decide per-batch where to run each block
            cpu_amx = should_offload_amx_to_cpu()
            cpu_moe = should_offload_moe_to_cpu()

            optimizer.zero_grad()
            preds = model(batch_x, use_cpu_amx=cpu_amx, use_cpu_moe=cpu_moe)
            loss = criterion(preds, batch_y.to('cpu'))
            loss.backward()

            # 1. Update CPU-resident parameters
            optimizer.step()

            # 2. Synchronize the updated CPU weights to the GPU copy
            for p_cpu, p_gpu in zip(model.moe_cpu.parameters(),
                                    model.moe_gpu.parameters()):
                p_gpu.data.copy_(p_cpu.data.to('cuda:0'))

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")


def main():
    # Example parameters
    in_dim = 128
    hidden_dim = 256
    out_dim = 64
    num_experts = 8
    batch_size = 32
    
    # Create model components
    amx_cpu = AMXModule(in_dim, hidden_dim).cpu()
    moe_cpu = MoEModule(hidden_dim, out_dim, num_experts).cpu()
    moe_gpu = copy.deepcopy(moe_cpu).to('cuda:0')
    
    # Create hybrid model
    model = HybridModel(amx_cpu, moe_cpu, moe_gpu)
    
    # Create dummy dataloader
    # In a real application, you'd use your actual dataset
    x = torch.randn(100, in_dim)
    y = torch.randn(100, out_dim)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True  # Enable pinned memory for faster CPU->GPU transfers
    )
    
    # Train the model
    train_hybrid_model(model, dataloader)


if __name__ == "__main__":
    main()
