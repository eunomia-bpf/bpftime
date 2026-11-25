#include <cstdint>

extern "C" __attribute__((noinline)) uint64_t
__bpftime_test_stripped_back_trace__func5(uint64_t x)
{
	asm volatile("" : "+r"(x));
	return x + 5;
}

extern "C" __attribute__((noinline)) uint64_t
__bpftime_test_stripped_back_trace__func4(uint64_t x)
{
	return __bpftime_test_stripped_back_trace__func5(x) + 4;
}

extern "C" __attribute__((noinline)) uint64_t
__bpftime_test_stripped_back_trace__func3(uint64_t x)
{
	return __bpftime_test_stripped_back_trace__func4(x) + 3;
}

extern "C" __attribute__((noinline)) uint64_t
__bpftime_test_stripped_back_trace__func2(uint64_t x)
{
	return __bpftime_test_stripped_back_trace__func3(x) + 2;
}

extern "C" __attribute__((noinline)) uint64_t
__bpftime_test_stripped_back_trace__func1(uint64_t x)
{
	return __bpftime_test_stripped_back_trace__func2(x) + 1;
}
