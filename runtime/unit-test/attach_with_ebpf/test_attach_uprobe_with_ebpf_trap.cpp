#include "bpftime_internal.h"
#if BPFTIME_HAVE_TRAP_UPROBE_ATTACH
#include "trap_uprobe_attach_impl.hpp"
#include "trap_attach_utils.hpp"
#include "spdlog/spdlog.h"
#include "unit-test/common_def.hpp"
#include <catch2/catch_test_macros.hpp>
#include "helper.hpp"
#include "bpf_attach_ctx.hpp"
#include "trap_attach_private_data.hpp"
#include "handler/handler_manager.hpp"
#include <cstring>
#include <memory>
#include <cstdlib>
#include <bpftime_shm_internal.hpp>
using namespace bpftime;

extern "C" __attribute__((noinline, noclone)) int
__bpftime_test_uprobe_with_ebpf_trap__my_function(int parm1, const char *str, char c)
{
	SPDLOG_INFO("Origin func: Args {}, {}, {}", parm1, str, c);
	return 35;
}

static const char *ebpf_prog_path = TOSTRING(EBPF_PROGRAM_PATH_UPROBE_TRAP);

TEST_CASE("Test probing internal functions with the trap backend")
{
	bpftime::runtime_config config;
	// DEFAULT_VM_NAME is "llvm" for builds with the LLVM JIT and "ubpf"
	// otherwise, so this keeps the previous behaviour where LLVM is present
	// and still works on builds without it.
	config.set_vm_name(DEFAULT_VM_NAME);
	REQUIRE(__bpftime_test_uprobe_with_ebpf_trap__my_function(1, "hello aaa",
							     'c') == 35);
	bpftime::shm_holder.global_shared_memory.set_runtime_config(std::move(config));
	std::unique_ptr<bpftime_object, decltype(&bpftime_object_close)> obj(
		bpftime_object_open(ebpf_prog_path),
		bpftime_object_close);
	REQUIRE(obj.get() != nullptr);
	SECTION("Test probing internal functions")
	{
		auto uretprobe_prog =
			create_bpftime_prog("my_function_uretprobe", obj.get());
		REQUIRE(uretprobe_prog->prepare_for_signal_execution() == 0);
		uint64_t ret = 1;
		pt_regs regs{};
		REQUIRE(uretprobe_prog->bpftime_prog_exec(&regs, sizeof(regs),
							  &ret) >= 0);
		REQUIRE(ret == 0);
		const auto expected_func_ip = reinterpret_cast<uintptr_t>(
			__bpftime_test_uprobe_with_ebpf_trap__my_function);
		attach::trap::trap_attach_impl man;
		auto uprobe_prog =
			create_bpftime_prog("my_function_uprobe", obj.get());
		REQUIRE(uprobe_prog->prepare_for_signal_execution() == 0);
		int entry_hits = 0, return_hits = 0;
		bool callback_failed = false;
		auto uprobe_hook_func = [&](const pt_regs &regs) {
			uint64_t ret = 0;
			int err = uprobe_prog->bpftime_prog_exec((void *)&regs, sizeof(regs), &ret);
			callback_failed |= err < 0 || ret != expected_func_ip;
			++entry_hits;
		};
		int id1 = man.create_uprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf_trap__my_function,
			uprobe_hook_func);
		REQUIRE(id1 >= 0);
		// Preparing a second link must preserve an already live program.
		REQUIRE(uprobe_prog->prepare_for_signal_execution() == 0);
		int id2 = man.create_uprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf_trap__my_function,
			uprobe_hook_func);
		REQUIRE(id2 >= 0);

		int id3 = man.create_uretprobe_at(
			(void *)__bpftime_test_uprobe_with_ebpf_trap__my_function,
			[&](const pt_regs &regs) {
				uint64_t ret = 0;
				int err = uretprobe_prog->bpftime_prog_exec((void *)&regs, sizeof(regs), &ret);
				callback_failed |= err < 0 || ret != expected_func_ip;
				++return_hits;
			});
		REQUIRE(id3 >= 0);
		REQUIRE(__bpftime_test_uprobe_with_ebpf_trap__my_function(
				1, "hello aaa", 'c') == 35);
		REQUIRE_FALSE(callback_failed);
		REQUIRE(entry_hits == 2);
		REQUIRE(return_hits == 1);
		REQUIRE(man.detach_by_id(id1) >= 0);
		REQUIRE(man.detach_by_id(id2) >= 0);
		REQUIRE(man.detach_by_id(id3) >= 0);
		ret = 1;
		REQUIRE(uretprobe_prog->bpftime_prog_exec(&regs, sizeof(regs),
							  &ret) >= 0);
		REQUIRE(ret == 0);
	}

	SECTION("Test probing libc functions")
	{
		attach::trap::trap_attach_impl man;
		auto strdup_addr = attach::trap::find_function_addr_by_name("strdup");
		REQUIRE(strdup_addr != nullptr);
		auto uprobe_prog =
			create_bpftime_prog("strdup_uprobe", obj.get());
		REQUIRE(uprobe_prog->prepare_for_signal_execution() == 0);
		int entry_hits = 0, return_hits = 0;
		bool callback_failed = false;
		int id1 = man.create_uprobe_at(
			strdup_addr, [&](const pt_regs &regs) {
				uint64_t ret = 1;
				int err = uprobe_prog->bpftime_prog_exec((void *)&regs, sizeof(regs), &ret);
				callback_failed |= err < 0 || ret != 0;
				++entry_hits;
			});
		REQUIRE(id1 >= 0);
		auto uretprobe_prog =
			create_bpftime_prog("strdup_uretprobe", obj.get());
		REQUIRE(uretprobe_prog->prepare_for_signal_execution() == 0);
		int id2 = man.create_uretprobe_at(
			strdup_addr, [&](const pt_regs &regs) {
				uint64_t ret = 1;
				int err = uretprobe_prog->bpftime_prog_exec((void *)&regs, sizeof(regs), &ret);
				callback_failed |= err < 0 || ret != 0;
				++return_hits;
			});
		REQUIRE(id2 >= 0);
		char *duped_str = strdup("aabbccdd");
		REQUIRE(std::string(duped_str) == "aabbccdd");
		free(duped_str);
		REQUIRE_FALSE(callback_failed);
		REQUIRE(entry_hits == 1);
		REQUIRE(return_hits == 1);
		REQUIRE(man.detach_by_id(id1) >= 0);
		REQUIRE(man.detach_by_id(id2) >= 0);
	}
}
// A native callback's implementation is a caller contract. Runtime-managed
// BPF programs are checked before attach, including helper implementations
// registered under an otherwise safe helper ID.
static uint64_t unaudited_helper(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	return 123;
}

TEST_CASE("Trap BPF admission rejects unaudited helper implementations")
{
	runtime_config config;
	config.set_vm_name(DEFAULT_VM_NAME);
	for (int helper : { 1, 2, 6, 12, 27, 67, 173, 174, 999 }) {
		CAPTURE(helper);
		ebpf_inst code[] = {
			{ .code = 0x85, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = helper },
			{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
		};
		bpftime_prog prog(code, 2, "unaudited", config);
		REQUIRE(prog.bpftime_prog_register_raw_helper({
			(unsigned)helper, "unaudited", (void *)unaudited_helper }) == 0);
		REQUIRE(prog.bpftime_prog_load(false) == 0);
		REQUIRE(prog.prepare_for_signal_execution() == -ENOTSUP);
		// Ordinary execution retains its previous helper support.
		pt_regs regs{};
		uint64_t ret = 0;
		REQUIRE(prog.bpftime_prog_exec(&regs, sizeof(regs), &ret) == 0);
		REQUIRE(ret == 123);
	}
}

TEST_CASE("Trap runtime links reject unsafe programs before patching code")
{
	using namespace boost::interprocess;
	for (int helper : { 6, 67, 173 }) {
		CAPTURE(helper);
		const std::string name = "bpftime_trap_admission_" + std::to_string(getpid());
		::shm_remove remover{ std::string(name) };
		managed_shared_memory segment(create_only, name.c_str(), 1 << 20);
		REQUIRE(segment.get_segment_manager() != nullptr);
		auto *manager = segment.construct<handler_manager>("manager")(segment, 8);
		REQUIRE(manager != nullptr);
		ebpf_inst code[] = {
			{ .code = 0x85, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = helper },
			{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
		};
		REQUIRE(manager->set_handler(0, bpf_prog_handler(segment, code, 2,
			"admission", 0), segment) >= 0);
		bpf_perf_event_handler event(false, 0, -1, "", 0, segment);
		event.enable();
		REQUIRE(manager->set_handler(1, event, segment) >= 0);
		REQUIRE(manager->set_handler(2, bpf_link_handler(0, 1), segment) >= 0);
		auto *target = (void *)__bpftime_test_uprobe_with_ebpf_trap__my_function;
		uint8_t original[4];
		memcpy(original, target, sizeof(original));
		{
			bpf_attach_ctx ctx;
			auto impl = std::make_unique<attach::trap::trap_attach_impl>();
			auto *trap = impl.get();
			ctx.register_attach_impl({ attach::trap::ATTACH_UPROBE }, std::move(impl),
				[target](const std::string_view &, int &err)
					-> std::unique_ptr<attach::attach_private_data> {
					err = 0;
					auto data = std::make_unique<attach::trap::trap_attach_private_data>();
					data->addr = (uintptr_t)target;
					return data;
				});
			runtime_config config;
			config.set_vm_name(DEFAULT_VM_NAME);
			config.jit_enabled = false;
			config.enable_kernel_helper_group = true;
			REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config) == 0);
			REQUIRE(ctx.find_instantiated_prog(0) != nullptr);
			int attached = 0;
			trap->iterate_attaches([&](int, const void *, int) { ++attached; });
			if (helper == 173) {
				REQUIRE(attached == 1);
			} else {
				REQUIRE(attached == 0);
				REQUIRE(memcmp(original, target, sizeof(original)) == 0);
			}
			REQUIRE(__bpftime_test_uprobe_with_ebpf_trap__my_function(1, "hello", 'c') == 35);
		}
		REQUIRE(memcmp(original, target, sizeof(original)) == 0);
		manager->clear_all(segment);
		segment.destroy_ptr(manager);
	}
}

TEST_CASE("Trap BPF execution cannot lazily compile an unprepared program")
{
	runtime_config config;
	config.set_vm_name(DEFAULT_VM_NAME);
	ebpf_inst code[] = {
		{ .code = 0xb7, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 123 },
		{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
	};
	bpftime_prog prog(code, 2, "not_prepared", config);
	REQUIRE(prog.bpftime_prog_load(false) == 0);
	attach::trap::trap_attach_impl man;
	int error = 0, hits = 0;
	const int id = man.create_uprobe_at(
		(void *)__bpftime_test_uprobe_with_ebpf_trap__my_function,
		[&](const pt_regs &regs) {
			uint64_t ret = 0;
			error = prog.bpftime_prog_exec((void *)&regs, sizeof(regs), &ret);
			++hits;
		});
	REQUIRE(id >= 0);
	REQUIRE(__bpftime_test_uprobe_with_ebpf_trap__my_function(1, "hello", 'c') == 35);
	REQUIRE(hits == 1);
	REQUIRE(error == -ENOTSUP);
	REQUIRE(man.detach_by_id(id) == 0);
}
#endif
