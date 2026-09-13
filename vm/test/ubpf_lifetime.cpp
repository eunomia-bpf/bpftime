#include <compat_ubpf.hpp>
#include <ebpf-vm.h>
#include <ebpf_inst.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<bpftime::vm::ubpf::bpftime_ubpf_vm>);
static_assert(!std::is_copy_assignable_v<bpftime::vm::ubpf::bpftime_ubpf_vm>);

static int created;
static int destroyed;

extern "C" {
struct ubpf_vm;
ubpf_vm *__real_ubpf_create();
void __real_ubpf_destroy(ubpf_vm *vm);

ubpf_vm *__wrap_ubpf_create()
{
	auto *vm = __real_ubpf_create();
	if (vm)
		++created;
	return vm;
}

void __wrap_ubpf_destroy(ubpf_vm *vm)
{
	++destroyed;
	__real_ubpf_destroy(vm);
}
}

// Keep checks active in Release builds as well.
static void check(bool condition, const char *message)
{
	if (!condition) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(EXIT_FAILURE);
	}
}

int main()
{
	const ebpf_inst code[] = {
		{ EBPF_OP_MOV64_IMM, 0, 0, 0, 42 },
		{ EBPF_OP_EXIT, 0, 0, 0, 0 },
	};
	// Empty, loaded, explicitly unloaded, rejected code, and JIT-compiled.
	for (int state = 0; state < 5; ++state) {
		auto *vm = ebpf_create("ubpf");
		check(created == state + 1, "Expected one backend allocation");
		char *error = nullptr;
		if (state == 3) {
			const ebpf_inst invalid = { 0xff, 0, 0, 0, 0 };
			check(ebpf_load(vm, &invalid, sizeof(invalid), &error) <
				      0,
			      "Invalid code should be rejected");
			std::free(error);
		} else if (state != 0) {
			check(ebpf_load(vm, code, sizeof(code), &error) == 0,
			      "Failed to load program");
			uint64_t result = 0;
			check(ebpf_exec(vm, nullptr, 0, &result) == 0 &&
				      result == 42,
			      "Unexpected interpreter result");
			if (state == 2)
				ebpf_unload_code(vm);
#if defined(__x86_64__) || defined(__aarch64__)
			if (state == 4) {
				auto jit = ebpf_compile(vm, &error);
				check(jit != nullptr,
				      "Failed to compile program");
				check(jit(nullptr, 0) == 42,
				      "Unexpected JIT result");
			}
#endif
		}
		ebpf_destroy(vm);
		check(destroyed == created,
		      "Backend must be destroyed exactly once per VM");
	}
}
