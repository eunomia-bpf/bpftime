// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
// Retain this definition during device-only compilation. The IR linker
// externalizes it before resolving the eBPF helper declaration.
extern "C" __device__ __attribute__((used)) unsigned long long
_bpf_helper_ext_0506(unsigned long long str, unsigned long long,
		     unsigned long long, unsigned long long, unsigned long long)
{
	return printf("%s\n", reinterpret_cast<const char *>(str));
}
