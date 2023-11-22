#include <catch2/catch_test_macros.hpp>
#include "helper.hpp"
#include <spdlog/spdlog.h>
#include <unit-test/common_def.hpp>
#include <attach/attach_manager/frida_attach_manager.hpp>
using namespace bpftime;

// This is the original function to hook.
__attribute__((__noinline__, optnone)) extern "C" int
__bpftime_attach_filter_with_ebpf__my_function(const char *str, char c,
					       long long parm1)
{
	asm("");
	// buggy code: not check str is NULL
	int i = str[0];
	spdlog::info("origin func: Args: {}, {}, {}", str, c, i);
	return (int)parm1;
}

extern "C" uint64_t __bpftime_attach_filter_with_ebpf__test_pass_param(
	uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t, uint64_t)
{
	char *str = (char *)arg1;
	char c = (char)arg2;
	long long param1 = (long long)arg3;
	// "hello aaa", 'c', 182
	REQUIRE(strcmp(str, "hello aaa") == 0);
	REQUIRE(c == 'c');
	REQUIRE(param1 == 182);
	return 0;
}

static const char *ebpf_prog_path = TOSTRING(EBPF_PROGRAM_PATH_FILTER);

TEST_CASE("Test attaching filter program with ebpf, and reverting")
{
	REQUIRE(__bpftime_attach_filter_with_ebpf__my_function("hello aaa", 'c',
							       182) == 182);
	std::unique_ptr<bpftime_object, decltype(&bpftime_object_close)> obj(
		bpftime_object_open(ebpf_prog_path), bpftime_object_close);
	REQUIRE(obj.get() != nullptr);
	frida_attach_manager man;
	auto prog = bpftime_object__next_program(obj.get(), nullptr);
	REQUIRE(prog != nullptr);
	REQUIRE(bpftime_helper_group::get_ffi_helper_group()
			.add_helper_group_to_prog(prog) >= 0);
	REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog) >= 0);
	bpftime_helper_info info = {
		.index = 4097,
		.name = "test_pass_param",
		.fn = (void *)__bpftime_attach_filter_with_ebpf__test_pass_param,
	};
	REQUIRE(prog->bpftime_prog_register_raw_helper(info) >= 0);
	REQUIRE(prog->bpftime_prog_load(false) >= 0);
	int id = man.attach_filter_at(
		(void *)__bpftime_attach_filter_with_ebpf__my_function,
		[=](const pt_regs &regs) -> bool {
			uint64_t ret;
			REQUIRE(prog->bpftime_prog_exec((void *)&regs,
							sizeof(regs),
							&ret) >= 0);
			return !ret;
		});
	REQUIRE(id >= 0);
	REQUIRE(__bpftime_attach_filter_with_ebpf__my_function("hello aaa", 'c',
							       182) == 182);
	REQUIRE(__bpftime_attach_filter_with_ebpf__my_function(nullptr, 'c',
							       1) == -22);
	SECTION("Destroy with id")
	{
		REQUIRE(man.destroy_attach(id) >= 0);
	}
	SECTION("Destroy with address")
	{
		REQUIRE(man.destroy_attach_by_func_addr((
				void *)__bpftime_attach_filter_with_ebpf__my_function) >=
			0);
	}
	REQUIRE(__bpftime_attach_filter_with_ebpf__my_function("hello aaa", 'c',
							       182) == 182);
}
