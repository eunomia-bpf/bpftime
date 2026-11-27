// SPDX-License-Identifier: MIT
// The traced workload for the Darwin preload-constructor test. It calls an
// exported function in a separate dylib a known number of times; the injected
// agent (DYLD_INSERT_LIBRARIES) arms a uprobe on that function and increments
// a shared-memory array map on every entry.
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unistd.h>

extern "C" uint64_t bpftime_macos_demo_compute(uint64_t a, uint64_t b);

int main(int argc, char **argv)
{
	uint32_t calls = argc > 1 ? static_cast<uint32_t>(std::stoul(argv[1]))
				  : 50u;
	uint64_t acc = 0;
	for (uint32_t i = 0; i < calls; ++i)
		acc += bpftime_macos_demo_compute(i, 1);
	printf("preload_victim pid=%d calls=%u acc=%llu\n", getpid(), calls,
	       static_cast<unsigned long long>(acc));
	return 0;
}
