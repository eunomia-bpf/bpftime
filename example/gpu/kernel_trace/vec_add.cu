#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

__constant__ int d_N;

// Dummy hook stub: when bpftime CUDA attach is enabled, PTX passes
// will retarget calls to this function to the eBPF-generated probe
// function for this kernel.
__device__ __noinline__ void __bpftime_cuda__kernel_trace()
{
	// Intentionally left almost empty to keep baseline overhead minimal
}

__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// User-visible hook point: this call will be rewritten at PTX level
	// to invoke the eBPF probe when bpftime is attached.
	__bpftime_cuda__kernel_trace();

	if (idx < d_N)
		C[idx] = A[idx] + B[idx];
}

int main()
{
	const int N = 1 << 12;
	size_t bytes = N * sizeof(float);

	std::vector<float> h_A(N), h_B(N), h_C(N);
	for (int i = 0; i < N; i++) {
		h_A[i] = i * 1.0f;
		h_B[i] = i * 3.0f;
	}
	cudaMemcpyToSymbol(d_N, &N, sizeof(N));

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	const int threads_per_block = 128;
	const int blocks = (N + threads_per_block - 1) / threads_per_block;

	while (true) {
		vectorAdd<<<blocks, threads_per_block>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C.data(), d_C, sizeof(float) * 2,
			   cudaMemcpyDeviceToHost);
		printf("vectorAdd => C[0]=%.1f C[1]=%.1f\n", h_C[0], h_C[1]);
		sleep(1);
	}

	return 0;
}
