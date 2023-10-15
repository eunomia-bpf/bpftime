#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include "bpf_progs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm_jit_context.h"
#include "ebpf-vm.h"

using namespace llvm;

struct ebpf_inst;

#define JIT_TEST_UBPF 1

#define TEST_BPF_CODE test_1
#define TEST_BPF_SIZE (sizeof(TEST_BPF_CODE) - 1)

typedef unsigned int (*kernel_fn)(const void *ctx, const struct ebpf_inst *insn);

char *errmsg;
struct mem {
	uint64_t val;
};

uint64_t ffi_print_func(uint64_t a, uint64_t _b, uint64_t _c, uint64_t _d,
			uint64_t _e)
{
	std::cout << (const char *)a << std::endl;
	return 0;
}
uint64_t ffi_add_func(uint64_t a, uint64_t b, uint64_t _c, uint64_t _d,
		      uint64_t _e)
{
	return a + b;
}

uint64_t ffi_print_integer(uint64_t a, uint64_t b, uint64_t _c, uint64_t _d,
			   uint64_t _e)
{
	std::cout << a << " -> " << b << " | " << std::endl;
	return 0;
}

uint8_t bpf_mem[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };

int main(int argc, char *argv[])
{
	// Initialize LLVM.
	uint64_t res = 0;
	struct ebpf_vm *vm = NULL;
	char *errmsg;
	
	InitLLVM X(argc, argv);

	InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();

	cl::ParseCommandLineOptions(argc, argv, "HowToUseLLJIT");

	vm = ebpf_create();
	if (!vm) {
		fprintf(stderr, "Failed to create VM\n");
		return 1;
	}
	res = ebpf_load(vm, TEST_BPF_CODE, TEST_BPF_SIZE, &errmsg);
	if (res) {
		fprintf(stderr, "Failed to load: %s\n", errmsg);
		return 1;
	}
	ebpf_register(vm, 2, "print", (void *)ffi_print_func);
	ebpf_register(vm, 3, "add", (void *)ffi_add_func);
	ebpf_register(vm, 4, "print_integer", (void *)ffi_print_integer);
	printf("code len: %zd\n", TEST_BPF_SIZE);
	auto func = ebpf_compile(vm, &errmsg);
	if (!func) {
		fprintf(stderr, "Failed to compile: %s\n", errmsg);
		return 1;
	}
	int err = ebpf_exec(vm, &bpf_mem, sizeof(bpf_mem), &res);
	if (err != 0) {
		fprintf(stderr, "Failed to exec: %s\n", errmsg);
		return 1;
	}
	printf("res = %" PRIu64 "\n", res);
	return 0;
}
