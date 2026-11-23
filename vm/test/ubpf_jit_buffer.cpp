#include <compat_ubpf.hpp>
#include <ebpf_inst.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Keep checks active in Release builds as well.
static void check(bool condition, const char *message)
{
	if (!condition) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(EXIT_FAILURE);
	}
}

// A straight-line program whose native image overflows uBPF's default
// 64 KiB JIT scratch buffer. Each MOV64_IMM expands to several native
// instructions, so several thousand of them produce a > 64 KiB image.
static std::vector<ebpf_inst> make_large_program(size_t mov_count)
{
	std::vector<ebpf_inst> insns;
	insns.reserve(mov_count + 1);
	for (size_t i = 0; i < mov_count; ++i) {
		insns.push_back(
			{ EBPF_OP_MOV64_IMM, 0, 0, 0, static_cast<int>(i) });
	}
	insns.push_back({ EBPF_OP_EXIT, 0, 0, 0, 0 });
	return insns;
}

int main()
{
#if defined(__x86_64__) || defined(__aarch64__)
	using bpftime::vm::ubpf::bpftime_ubpf_vm;

	// Small programs keep using the default buffer: no behavior change.
	{
		bpftime_ubpf_vm vm;
		const ebpf_inst small[] = {
			{ EBPF_OP_MOV64_IMM, 0, 0, 0, 7 },
			{ EBPF_OP_EXIT, 0, 0, 0, 0 },
		};
		check(vm.load_code(small, sizeof(small)) == 0,
		      "small program failed to load");
		auto jit = vm.compile();
		check(jit.has_value(), "small program failed to JIT compile");
		check((*jit)(nullptr, 0) == 7,
		      "small program returned an unexpected JIT result");
	}

	// Large programs: without buffer growth uBPF reports
	// "Target buffer too small"; the compat layer must grow and succeed.
	{
		bpftime_ubpf_vm vm;
		auto insns = make_large_program(8000);
		check(vm.load_code(insns.data(),
				   insns.size() * sizeof(ebpf_inst)) == 0,
		      "large program failed to load");
		auto jit = vm.compile();
		if (!jit.has_value()) {
			check(false,
			      ("large program JIT should grow past 64 KiB: " +
			       vm.get_error_message())
				      .c_str());
		}
		check((*jit)(nullptr, 0) == insns.size() - 2,
		      "large program returned an unexpected JIT result");
	}
#else
	std::printf("uBPF JIT is only supported on x86_64/aarch64; skipped\n");
#endif
	return 0;
}
