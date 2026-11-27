// SPDX-License-Identifier: MIT
// Manual GPU regression: both lock contention and a missing listener must
// release the kernel after the deadline, without recycling the request slot.
#include "../device_helpers.hip"
#include <chrono>
#include <cstdio>
#include <cstring>

__device__ unsigned long long channel_address;
extern "C" __device__ unsigned long long bpftime_rocm_channel_address()
{
	return channel_address;
}
__global__ void timeout_probe(unsigned long long *out)
{
	uint32_t key = 0;
	uint64_t value = 1;
	out[0] = _bpf_helper_ext_0001(1, (uint64_t)&key, 0, 0, 0);
	out[1] =
		_bpf_helper_ext_0002(1, (uint64_t)&key, (uint64_t)&value, 0, 0);
}
int main()
{
	rocm_channel_data *channel;
	unsigned long long *output;
	int rate = 0;
	if (hipHostMalloc(&channel, sizeof(*channel),
			  hipHostMallocMapped | hipHostMallocCoherent) !=
		    hipSuccess ||
	    hipMallocManaged(&output, 2 * sizeof(*output)) != hipSuccess ||
	    hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0) !=
		    hipSuccess)
		return 1;
	auto address = (unsigned long long)channel;
	if (hipMemcpyToSymbol(HIP_SYMBOL(channel_address), &address,
			      sizeof(address)) != hipSuccess)
		return 1;
	for (unsigned held : { 0u, 1u }) {
		std::memset(channel, 0, sizeof(*channel));
		channel->lock = held;
		channel->map_count = 1;
		channel->value_sizes[0] = 8;
		channel->timeout_ticks = uint64_t(rate) * 100; // 100 ms
		auto start = std::chrono::steady_clock::now();
		hipLaunchKernelGGL(timeout_probe, dim3(1), dim3(1), 0, 0,
				   output);
		if (hipDeviceSynchronize() != hipSuccess)
			return 1;
		const double elapsed =
			std::chrono::duration<double>(
				std::chrono::steady_clock::now() - start)
				.count();
		if (output[0] != 0 || int64_t(output[1]) != -110 ||
		    !channel->stopped || elapsed > 2 ||
		    (held ? channel->state != rocm_idle :
			    channel->state != rocm_request))
			return 1;
		std::printf(
			"channel timeout: held=%u seconds=%.3f lookup=NULL update=-ETIMEDOUT stopped=1\n",
			held, elapsed);
	}
	return hipFree(output) == hipSuccess &&
			       hipHostFree(channel) == hipSuccess ?
		       0 :
		       1;
}
