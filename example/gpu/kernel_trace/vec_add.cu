#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
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
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

	const int threads_per_block = 128;
	const int blocks = (N + threads_per_block - 1) / threads_per_block;

	while (true) {
		vectorAdd<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C.data(), d_C, sizeof(float) * 2,
			   cudaMemcpyDeviceToHost);
		printf("vectorAdd => C[0]=%.1f C[1]=%.1f\n", h_C[0], h_C[1]);
		sleep(1);
	}

	return 0;
}
