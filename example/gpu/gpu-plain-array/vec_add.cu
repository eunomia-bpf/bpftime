#include <cuda_runtime.h>
#include <unistd.h>

__constant__ int d_N;

__global__ void vectorAdd(const float *A, const float *B, float *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	while (idx < d_N) {
		C[idx] = A[idx] + B[idx];
		idx += blockDim.x * gridDim.x;
	}
}

int main()
{
	const int h_N = 1 << 10;
	cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));

	size_t bytes = h_N * sizeof(float);
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	while (1) {
		cudaMemset(d_C, 0, bytes);
		vectorAdd<<<8, 64>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();
		sleep(1);
	}
	return 0;
}


