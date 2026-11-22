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
#include "spdlog/spdlog.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <memory>
#include <string>
#include <unistd.h>

using namespace bpftime;
using namespace boost::interprocess;

// A handler that cannot be instantiated (here: a uprobe whose attach type
// has no registered implementation) used to be reported by one DEBUG line
// per handler while init_attach_ctx_from_handlers still returned 0, so a
// probe that never got attached looked like a successful session. The scan
// now keeps the failing ids and errors, warns once, and exposes them.
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
	bpf_attach_ctx ctx;
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config) == 0);
	REQUIRE(ctx.find_instantiated_prog(0) != nullptr);

	auto failed = ctx.failed_instantiated_handlers();
	REQUIRE(failed.count(1) == 1);
	REQUIRE(failed.at(1) == -ENOENT);
	// The link fails because its attach target could not be instantiated.
	REQUIRE(failed.count(2) == 1);
	REQUIRE(failed.at(2) < 0);

	// A scan with every handler instantiated clears the record.
	manager->clear_id_at(1, segment);
	manager->clear_id_at(2, segment);
	ctx.reset_instantiated_state();
	REQUIRE(ctx.failed_instantiated_handlers().empty());
	REQUIRE(ctx.init_attach_ctx_from_handlers(manager, config) == 0);
	REQUIRE(ctx.failed_instantiated_handlers().empty());

	ctx.destroy_all_attach_links();
	ctx.reset_instantiated_state();
	manager->clear_all(segment);
	segment.destroy_ptr(manager);
}