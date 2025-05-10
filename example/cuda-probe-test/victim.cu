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

/*
nvcc -x cu -cuda victim.cu -o victim.cpp
python filter_hashtag.py
g++ victim-new.cpp -Wall -L /usr/local/cuda-12.6/lib64 -lcudart -o victim -g
 */

__constant__ int const_integer;

__global__ void sumArray(uint64_t *input, uint64_t *output, size_t size)
{
	for (size_t i = 0; i < size; i++)
		*output += input[i];
	printf("From device side: sum = %lu, const integer = %d\n",
	       (unsigned long)*output, const_integer);
}

__global__ void sumArraySqr(uint64_t *input, uint64_t *output, size_t size)
{
	for (size_t i = 0; i < size; i++)
		*output += input[i] * input[i];
	printf("From device side: sumSqr = %lu\n", (unsigned long)*output);
}

constexpr size_t ARR_SIZE = 1000;

int main()
{
	int value = 1;
	int err = cudaMemcpyToSymbol(const_integer, &value, sizeof(value));
	std::cout<<"copy error = "<<(int)err<<std::endl;
	std::vector<uint64_t> arr;
	for (size_t i = 1; i <= ARR_SIZE; i++) {
		arr.push_back(i);
	}
	auto data_size = sizeof(arr[0]) * arr.size();

	uint64_t *d_input, *sum_output, *sqr_output;
	cudaMalloc(&d_input, data_size);
	cudaMalloc(&sum_output, sizeof(arr[0]));
	cudaMalloc(&sqr_output, sizeof(arr[0]));

	cudaMemcpy(d_input, arr.data(), data_size, cudaMemcpyHostToDevice);
	while (true) {
		sumArray<<<1, 1, 1>>>(d_input, sum_output, arr.size());
		sumArraySqr<<<1, 1, 1>>>(d_input, sqr_output, arr.size());

		uint64_t host_sum, host_sqr;
		cudaMemcpy(&host_sum, sum_output, sizeof(arr[0]),
			   cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_sqr, sqr_output, sizeof(arr[0]),
			   cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		std::cout << "Sum is " << host_sum << ", sqr is " << host_sqr
			  << std::endl;
		sleep(1);
	}
	cudaFree(d_input);
	cudaFree(sum_output);

	return 0;
}
