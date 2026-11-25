// SPDX-License-Identifier: MIT
// The traced target: a deliberately simple, never-inlined exported function
// living in its own shared library (.dylib on macOS, .so on Linux). Keeping it
// out of the main executable makes the dladdr module/offset resolution and the
// Frida inline hook exercise the same path on both platforms.
#include <cstdint>

#if defined(_WIN32)
#define BPFTIME_DEMO_EXPORT __declspec(dllexport)
#else
#define BPFTIME_DEMO_EXPORT __attribute__((visibility("default")))
#endif

extern "C" BPFTIME_DEMO_EXPORT __attribute__((noinline)) uint64_t
bpftime_macos_demo_compute(uint64_t a, uint64_t b)
{
	// Prevent the compiler from folding the body into the caller regardless
	// of optimization level.
	asm volatile("" : "+r"(a), "+r"(b) : : "memory");
	return a * 10 + b;
}
