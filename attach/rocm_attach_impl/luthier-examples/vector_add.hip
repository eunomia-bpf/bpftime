// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t status, const char *operation)
{
	if (status != hipSuccess)
		throw std::runtime_error(std::string(operation) + ": " +
					 hipGetErrorString(status));
}

// No Luthier includes, hook calls, instrumentation IR, or device globals.
// The early return exercises a partial final workgroup.
extern "C" __global__ void bpftime_luthier_vector_add(const uint32_t *a,
						      const uint32_t *b,
						      uint32_t *out, uint32_t n)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	const uint32_t value = a[i] + b[i];
	out[i] = (i & 1) ? value * 3u + 1u : value ^ 0x1234u;
}

struct device_buffer {
	uint32_t *ptr = nullptr;
	explicit device_buffer(size_t bytes)
	{
		check(hipMalloc(&ptr, bytes), "hipMalloc");
	}
	~device_buffer()
	{
		if (ptr)
			(void)hipFree(ptr);
	}
	device_buffer(const device_buffer &) = delete;
	device_buffer &operator=(const device_buffer &) = delete;
};

int main()
{
	try {
		int count = 0;
		check(hipGetDeviceCount(&count), "hipGetDeviceCount");
		if (count == 0)
			throw std::runtime_error("No HIP device available");
		check(hipSetDevice(0), "hipSetDevice");
		hipDeviceProp_t properties{};
		check(hipGetDeviceProperties(&properties, 0),
		      "hipGetDeviceProperties");
		uint64_t launches = 0, waves = 0, checksum = 0;
		for (uint32_t n : { 1003u, 4097u }) {
			for (uint32_t block : { 64u, 128u }) {
				std::vector<uint32_t> a(n), b(n), out(n);
				for (uint32_t i = 0; i < n; ++i) {
					a[i] = i * 17u + 3u;
					b[i] = i * 5u + 9u;
				}
				const size_t bytes = n * sizeof(uint32_t);
				device_buffer da(bytes), db(bytes), dout(bytes);
				check(hipMemcpy(da.ptr, a.data(), bytes,
						hipMemcpyHostToDevice),
				      "copy a");
				check(hipMemcpy(db.ptr, b.data(), bytes,
						hipMemcpyHostToDevice),
				      "copy b");
				const uint32_t grid = (n + block - 1) / block;
				for (unsigned repeat = 0; repeat < 3;
				     ++repeat) {
					std::fill(out.begin(), out.end(), 0u);
					check(hipMemcpy(dout.ptr, out.data(),
							bytes,
							hipMemcpyHostToDevice),
					      "clear output");
					hipLaunchKernelGGL(
						bpftime_luthier_vector_add,
						dim3(grid), dim3(block), 0,
						nullptr, da.ptr, db.ptr,
						dout.ptr, n);
					check(hipGetLastError(),
					      "launch vector_add");
					check(hipDeviceSynchronize(),
					      "synchronize vector_add");
					check(hipMemcpy(out.data(), dout.ptr,
							bytes,
							hipMemcpyDeviceToHost),
					      "copy output");
					for (uint32_t i = 0; i < n; ++i) {
						const uint32_t sum =
							a[i] + b[i];
						const uint32_t expected =
							(i & 1) ?
								sum * 3u + 1u :
								sum ^ 0x1234u;
						if (out[i] != expected)
							throw std::runtime_error(
								"Incorrect result at element " +
								std::to_string(
									i));
						checksum += out[i];
					}
					++launches;
					waves += uint64_t(grid) *
						 ((block + properties.warpSize -
						   1) /
						  properties.warpSize);
				}
			}
		}
		std::cout << "{\"status\":\"pass\",\"launches\":" << launches
			  << ",\"waves\":" << waves
			  << ",\"wave_size\":" << properties.warpSize
			  << ",\"checksum\":" << checksum << ",\"arch\":\""
			  << properties.gcnArchName << "\"}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
