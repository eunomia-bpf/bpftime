/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_internal.h"
#include <bpftime_shm_internal.hpp>
#include "bpf_attach_ctx.hpp"
#include "ebpf_inst.h"
#include "handler/handler_manager.hpp"
#include "handler/link_handler.hpp"
#include "common_def.hpp"
#include "handler/perf_event_handler.hpp"
#include "handler/prog_handler.hpp"
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace bpftime;
using namespace boost::interprocess;

namespace {
// Restores the process-wide default logger even when a REQUIRE aborts the
// test with an exception, so later tests never log through a dead sink.
struct logger_restore_guard {
	std::shared_ptr<spdlog::logger> previous;
	explicit logger_restore_guard(
		std::shared_ptr<spdlog::logger> logger)
		: previous(std::move(logger))
	{
	}
	~logger_restore_guard()
	{
		spdlog::set_default_logger(previous);
	}
};
} // namespace

// A handler that cannot be instantiated (here: a uprobe whose attach type
// has no registered implementation) used to be reported by one DEBUG line
// per handler while init_attach_ctx_from_handlers still returned 0, so a
// probe that never got attached looked like a successful session. The scan
// now returns the distinct failed handler count through an optional output
// argument and warns once, with the ids, only when the scan is stable.
TEST_CASE("Failed handler instantiation is reported, not swallowed")
{
	const std::string name =
		"bpftime_attach_failures_" + std::to_string(getpid());
	::shm_remove remover{ std::string(name) };
	managed_shared_memory segment(create_only, name.c_str(), 1 << 20);
	REQUIRE(segment.get_segment_manager() != nullptr);
	auto *manager = segment.construct<handler_manager>("manager")(segment, 8);
	REQUIRE(manager != nullptr);

	const ebpf_inst code[] = {
		{ .code = 0xb7, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
		{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
	};
	REQUIRE(manager->set_handler(0,
				     bpf_prog_handler(segment, code, 2, "prog", 0),
				     segment) == 0);
	// A uprobe perf event with no attach implementation registered on the
	// context: instantiation fails with -ENOENT.
	bpf_perf_event_handler event(false, 0, -1, "", 0, segment);
	event.enable();
	REQUIRE(manager->set_handler(1, event, segment) == 1);
	REQUIRE(manager->set_handler(2, bpf_link_handler(0, 1), segment) == 2);

	runtime_config config;
	// bpftime_prog_load reads the VM name from the global runtime config,
	// not from the config argument, so both have to name the VM.
	config.set_vm_name(DEFAULT_VM_NAME);
	runtime_config global_config;
	global_config.set_vm_name(DEFAULT_VM_NAME);
	bpftime::shm_holder.global_shared_memory.set_runtime_config(
		std::move(global_config));

	std::ostringstream logs;
	auto previous_logger = spdlog::default_logger();
	auto logger = std::make_shared<spdlog::logger>(
		"attach_failures",
		std::make_shared<spdlog::sinks::ostream_sink_mt>(logs));
	logger->set_level(spdlog::level::warn);
	logger->set_pattern("%v");
	spdlog::set_default_logger(logger);
	logger_restore_guard restore_logger(previous_logger);

	const std::string aggregate_warning =
		"bpftime: 2 handler(s) could not be instantiated: 1 (-2), 2 (-2)";
	auto count_aggregate_warnings = [&]() {
		const std::string text = logs.str();
		size_t count = 0;
		size_t pos = 0;
		while ((pos = text.find(aggregate_warning, pos)) !=
		       std::string::npos) {
			++count;
			pos += aggregate_warning.size();
		}
		return count;
	};

	bpf_attach_ctx ctx;
	// 1) The optional output argument receives the distinct failed
	// handler count, the good handler stays instantiated, and one
	// aggregate warning carries both ids and their error codes.
	int failed = -1;
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config,
						  &failed) == 0);
	REQUIRE(failed == 2);
	REQUIRE(ctx.find_instantiated_prog(0) != nullptr);
	REQUIRE(count_aggregate_warnings() == 1);

	// 2) After the failing handlers are removed and the context is
	// reset, a fresh scan reports zero failures and emits no new
	// aggregate warning.
	manager->clear_id_at(1, segment);
	manager->clear_id_at(2, segment);
	ctx.reset_instantiated_state();
	failed = -1;
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config,
						  &failed) == 0);
	REQUIRE(failed == 0);
	REQUIRE(count_aggregate_warnings() == 1);

	// 3) The default nullptr output argument keeps the call working.
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config) == 0);
	REQUIRE(count_aggregate_warnings() == 1);

	ctx.destroy_all_attach_links();
	ctx.reset_instantiated_state();
	manager->clear_all(segment);
	segment.destroy_ptr(manager);
}

#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
#include <nv_attach_private_data.hpp>

namespace {
// Custom attach types that do not collide with the bpf_event_type enum.
constexpr int TEST_GPU_REJECT_ATTACH_TYPE = 9101;
constexpr int TEST_GPU_PASS_ATTACH_TYPE = 9102;

struct test_stub_private_data final : attach::attach_private_data {
	std::string marker;
	int
	initialize_from_string(const std::string_view &sv) override
	{
		marker = std::string(sv);
		return 0;
	}
	std::string to_string() const override { return marker; }
};

// One stub backs both attach types: the "reject" entry returns the same
// error code the nv_attach_impl returns when STRICT verification fails,
// the other attaches successfully.
class test_gpu_reject_stub final : public attach::base_attach_impl {
      public:
	int successful_attaches = 0;
	int detach_calls = 0;

	int detach_by_id(int) override
	{
		++detach_calls;
		return 0;
	}
	int create_attach_with_ebpf_callback(
		attach::ebpf_run_callback &&,
		const attach::attach_private_data &private_data,
		int attach_type) override
	{
		if (private_data.to_string() == "reject")
			return attach::GPU_VERIFIER_REJECTED;
		++successful_attaches;
		return allocate_id();
	}
};
} // namespace

// STRICT-mode GPU verification rejection must fail only the rejected GPU
// handler: the scan returns 0 with the failure counted, sibling programs
// and links stay instantiated, and a stable re-scan neither destroys nor
// re-creates them.
TEST_CASE("GPU verifier rejection skips only the rejected handler")
{
	const std::string name =
		"bpftime_attach_gpu_reject_" + std::to_string(getpid());
	::shm_remove remover{ std::string(name) };
	managed_shared_memory segment(create_only, name.c_str(), 1 << 20);
	REQUIRE(segment.get_segment_manager() != nullptr);
	auto *manager = segment.construct<handler_manager>("manager")(
		segment, 16);
	REQUIRE(manager != nullptr);

	const ebpf_inst code[] = {
		{ .code = 0xb7, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
		{ .code = 0x95, .dst_reg = 0, .src_reg = 0, .off = 0, .imm = 0 },
	};
	REQUIRE(manager->set_handler(
			0, bpf_prog_handler(segment, code, 2, "prog_a", 0),
			segment) == 0);
	REQUIRE(manager->set_handler(
			3, bpf_prog_handler(segment, code, 2, "prog_b", 0),
			segment) == 3);
	// Rejected GPU link: perf event 1 + link 2.
	bpf_perf_event_handler rejected_event(
		TEST_GPU_REJECT_ATTACH_TYPE, "reject", segment);
	rejected_event.enable();
	REQUIRE(manager->set_handler(1, rejected_event, segment) == 1);
	REQUIRE(manager->set_handler(2, bpf_link_handler(0, 1), segment) == 2);
	// Healthy sibling link: perf event 4 + link 5.
	bpf_perf_event_handler healthy_event(TEST_GPU_PASS_ATTACH_TYPE, "ok",
					     segment);
	healthy_event.enable();
	REQUIRE(manager->set_handler(4, healthy_event, segment) == 4);
	REQUIRE(manager->set_handler(5, bpf_link_handler(3, 4), segment) == 5);

	runtime_config config;
	config.set_vm_name(DEFAULT_VM_NAME);
	runtime_config global_config;
	global_config.set_vm_name(DEFAULT_VM_NAME);
	bpftime::shm_holder.global_shared_memory.set_runtime_config(
		std::move(global_config));

	std::ostringstream logs;
	auto previous_logger = spdlog::default_logger();
	auto logger = std::make_shared<spdlog::logger>(
		"attach_gpu_reject",
		std::make_shared<spdlog::sinks::ostream_sink_mt>(logs));
	logger->set_level(spdlog::level::warn);
	logger->set_pattern("%v");
	spdlog::set_default_logger(logger);
	logger_restore_guard restore_logger(previous_logger);

	auto stub = std::make_unique<test_gpu_reject_stub>();
	auto *stub_ptr = stub.get();
	bpf_attach_ctx ctx;
	ctx.register_attach_impl(
		{ TEST_GPU_REJECT_ATTACH_TYPE, TEST_GPU_PASS_ATTACH_TYPE },
		std::move(stub),
		[](const std::string_view &sv, int &err)
			-> std::unique_ptr<attach::attach_private_data> {
			auto data = std::make_unique<
				test_stub_private_data>();
			err = data->initialize_from_string(sv);
			return data;
		});

	// 1) The scan succeeds overall, counts exactly one failed handler
	// (the rejected link), keeps both programs and the healthy link,
	// and logs the rejection explicitly instead of tearing the session
	// down.
	int failed = -1;
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config,
						  &failed) == 0);
	REQUIRE(failed == 1);
	REQUIRE(ctx.find_instantiated_prog(0) != nullptr);
	REQUIRE(ctx.find_instantiated_prog(3) != nullptr);
	REQUIRE(stub_ptr->successful_attaches == 1);
	REQUIRE(stub_ptr->detach_calls == 0);
	REQUIRE(logs.str().find("GPU verifier rejected handler 2") !=
		std::string::npos);
	REQUIRE(logs.str().find("2 (" + std::to_string(
					  attach::GPU_VERIFIER_REJECTED) +
				 ")") != std::string::npos);

	// 2) A stable re-scan keeps the healthy link attached: it is neither
	// detached nor attached a second time, and the rejection is still
	// counted once.
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config,
						  &failed) == 0);
	REQUIRE(failed == 1);
	REQUIRE(stub_ptr->successful_attaches == 1);
	REQUIRE(stub_ptr->detach_calls == 0);
	REQUIRE(ctx.find_instantiated_prog(0) != nullptr);
	REQUIRE(ctx.find_instantiated_prog(3) != nullptr);

	ctx.destroy_all_attach_links();
	ctx.reset_instantiated_state();
	manager->clear_all(segment);
	segment.destroy_ptr(manager);
}
#endif // BPFTIME_ENABLE_CUDA_ATTACH
