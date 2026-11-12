# How to test with faiss

## Get faiss and build it with GPU support
```
git clone --recursive https://github.com/facebookresearch/faiss

cd faiss
# NOTE:
# - Adjust `-DCUDAToolkit_ROOT=/usr/local/cuda-12.6` to match your CUDA installation path and version.
# - Set `-DCMAKE_CUDA_ARCHITECTURES="61"` to match your GPU architecture:
#     61 = Pascal, 75 = Turing, 80 = Ampere, 89 = Ada, etc.
cmake -DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCMAKE_CUDA_ARCHITECTURES="61" -DFAISS_ENABLE_ROCM=OFF -S . -B build -G Ninja
cmake --build build --config Debug --target demo_ivfpq_indexing_gpu

```

## Run bpftime

- Terminal 1, in `example/gpu/faiss-test`: `bpftime load ./threadhist`
- Terminal 2, in `faiss`: `bpftime start ./build/faiss/gpu/test/demo_ivfpq_indexing_gpu`

## Result

From the syscall server
```
16:01:39 
Thread 0: 236
Thread 1: 236
Thread 2: 236
Thread 3: 236
Thread 4: 236
Thread 5: 236
Thread 6: 236
```
