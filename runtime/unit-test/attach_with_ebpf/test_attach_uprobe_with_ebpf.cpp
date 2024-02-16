#include "bpftime_internal.h"
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/spdlog.h"
#include "unit-test/common_def.hpp"
#include <catch2/catch_test_macros.hpp>
#include "helper.hpp"
#include <memory>
#include <cstdlib>
#include <frida_uprobe.hpp>
using namespace bpftime;

extern "C" int
__bpftime_test_uprobe_with_ebpf__my_function(int parm1, const char *str, char c)
{
	SPDLOG_INFO("Origin func: Args {}, {}, {}", parm1, str, c);
	return 35;
}

static const char *ebpf_prog_path = TOSTRING(EBPF_PROGRAM_PATH_UPROBE);

TEST_CASE("Test probing internal functions")
{
	REQUIRE(__bpftime_test_uprobe_with_ebpf__my_function(1, "hello aaa",
							     'c') == 35);
	std::unique_ptr<bpftime_object, decltype(&bpftime_object_close)> obj(
		bpftime_object_open(ebpf_prog_path), bpftime_object_close);
	REQUIRE(obj.get() != nullptr);
	attach::frida_attach_impl man;
	SECTION("Test probing internal functions")
	{
		auto uprobe_prog =
			create_bpftime_prog("my_function_uprobe", obj.get());
		auto uprobe_hook_func = [=](const pt_regs &regs) {
			uint64_t ret;
			REQUIRE(uprobe_prog->bpftime_prog_exec((void *)&regs,
							       sizeof(regs),
							       &ret) >= 0);
		};
		int id1 = man.create_uprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf__my_function,
			uprobe_hook_func);
		REQUIRE(id1 >= 0);
		int id2 = man.create_uprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf__my_function,
			uprobe_hook_func);
		REQUIRE(id2 >= 0);

		auto uretprobe_prog =
			create_bpftime_prog("my_function_uretprobe", obj.get());
		// The only thing we could test is that if the ebpf program runs
		// successfully..
		int id3 = man.create_uretprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf__my_function,
			[=](const pt_regs &regs) {
				uint64_t ret;
				REQUIRE(uretprobe_prog->bpftime_prog_exec(
						(void *)&regs, sizeof(regs),
						&ret) >= 0);
			});
		REQUIRE(id3 >= 0);
		REQUIRE(__bpftime_test_uprobe_with_ebpf__my_function(
				1, "hello aaa", 'c') == 35);
		REQUIRE(man.detach_by_id(id1) >= 0);
		REQUIRE(man.detach_by_id(id2) >= 0);
		REQUIRE(man.detach_by_id(id3) >= 0);
	}
	
	SECTION("Test probing libc functions")
	{
		auto strdup_addr = attach::find_function_addr_by_name("strdup");
		REQUIRE(strdup_addr != nullptr);
		auto uprobe_prog =
			create_bpftime_prog("strdup_uprobe", obj.get());
		int id1 = man.create_uprobe_at(
			strdup_addr, [=](const pt_regs &regs) {
				uint64_t ret;
				REQUIRE(uprobe_prog->bpftime_prog_exec(
						(void *)&regs, sizeof(regs),
						&ret) >= 0);
			});
		REQUIRE(id1 >= 0);
		auto uretprobe_prog =
			create_bpftime_prog("strdup_uretprobe", obj.get());
		int id2 = man.create_uretprobe_at(
			strdup_addr, [=](const pt_regs &regs) {
				uint64_t ret;
				REQUIRE(uretprobe_prog->bpftime_prog_exec(
						(void *)&regs, sizeof(regs),
						&ret) >= 0);
			});
		REQUIRE(id2 >= 0);
		char *duped_str = strdup("aabbccdd");
		REQUIRE(std::string(duped_str) == "aabbccdd");
		free(duped_str);
		REQUIRE(man.detach_by_id(id1) >= 0);
		REQUIRE(man.detach_by_id(id2) >= 0);
	}
}
