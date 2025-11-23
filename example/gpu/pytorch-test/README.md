# How to test with pytorch

## Get pytorch
```
git clone --recursive https://github.com/pytorch/pytorch
```

## Compile pytorch with PTX included
```
uv venv
source ./.venv/bin/activate
uv pip install --group dev
uv pip install mkl-static mkl-include
USE_NCCL=0 USE_MPI=0 TORCH_CUDA_ARCH_LIST=6.1+PTX USE_XPU=0 USE_ROCM=0 REL_WITH_DEB_INFO=1 CMAKE_POLICY_VERSION_MINIMUM=3.5 CUDA_HOME=/usr/local/cuda-12.6 uv pip install --no-build-isolation -v -e .
```

## Run pytorch

Save the following program as `pytorch_test.py`

```python
import torch

data = torch.randint(0, 1000, (10,), dtype=torch.int32, device='cuda')

sorted_data = torch.sort(data)

print(f"Original: {data[:10]}")
print(f"Sorted: {sorted_data.values[:10]}")

```

In terminal 1, run `bpftime load ./threadhist` in this folder

In terminal 2, run `bpftime start python pytorch_test.py` (Must be in the uv venv created before)
