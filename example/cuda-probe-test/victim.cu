#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

__global__ void sumArray(uint64_t *input, uint64_t *output, size_t size)
{
	for (size_t i = 0; i < size; i++)
		*output += input[i];
	printf("From device side: sum = %lu\n", (unsigned long)*output);
}

constexpr size_t ARR_SIZE = 1000;

int main()
{
	std::vector<uint64_t> arr;
	for (size_t i = 1; i <= ARR_SIZE; i++) {
		arr.push_back(i);
	}
	auto data_size = sizeof(arr[0]) * arr.size();

	uint64_t *d_input, *d_output;
	cudaMalloc(&d_input, data_size);
	cudaMalloc(&d_output, sizeof(arr[0]));
	cudaMemcpy(d_input, arr.data(), data_size, cudaMemcpyHostToDevice);
	while (true) {
		sumArray<<<1, 1, 1>>>(d_input, d_output, arr.size());
		uint64_t host_sum;
		cudaMemcpy(&host_sum, d_output, sizeof(arr[0]),
			   cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::cout << "Sum is " << host_sum << std::endl;
		sleep(1);
	}
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}
