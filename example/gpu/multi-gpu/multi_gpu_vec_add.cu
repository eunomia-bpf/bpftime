// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
	do {                                                                   \
		cudaError_t error = (call);                                    \
		if (error != cudaSuccess) {                                    \
			fprintf(stderr, "%s failed: %s\n", #call,              \
				cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                              \
	} while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count)
		c[index] = a[index] + b[index];
}

struct device_data {
	float *a;
	float *b;
	float *c;
	cudaStream_t stream;
};

int main(int argc, char **argv)
{
	int available;
	CUDA_CHECK(cudaGetDeviceCount(&available));
	int device_count = argc > 1 ? atoi(argv[1]) : available;
	if (device_count < 1 || device_count > available) {
		fprintf(stderr, "Requested %d GPU(s), but %d available\n",
			device_count, available);
		return 1;
	}

	constexpr int count = 1 << 20;
	constexpr size_t bytes = count * sizeof(float);
	std::vector<device_data> devices(device_count);

	for (int device = 0; device < device_count; ++device) {
		auto &data = devices[device];
		CUDA_CHECK(cudaSetDevice(device));
		CUDA_CHECK(cudaStreamCreate(&data.stream));
		CUDA_CHECK(cudaMalloc(&data.a, bytes));
		CUDA_CHECK(cudaMalloc(&data.b, bytes));
		CUDA_CHECK(cudaMalloc(&data.c, bytes));
		CUDA_CHECK(cudaMemsetAsync(data.a, 0, bytes, data.stream));
		CUDA_CHECK(cudaMemsetAsync(data.b, 0, bytes, data.stream));
		vectorAdd<<<(count + 255) / 256, 256, 0, data.stream>>>(
			data.a, data.b, data.c, count);
		CUDA_CHECK(cudaGetLastError());
	}

	for (int device = 0; device < device_count; ++device) {
		auto &data = devices[device];
		CUDA_CHECK(cudaSetDevice(device));
		CUDA_CHECK(cudaStreamSynchronize(data.stream));
		CUDA_CHECK(cudaFree(data.a));
		CUDA_CHECK(cudaFree(data.b));
		CUDA_CHECK(cudaFree(data.c));
		CUDA_CHECK(cudaStreamDestroy(data.stream));
	}

	printf("vectorAdd completed on %d GPU(s)\n", device_count);
	return 0;
}
