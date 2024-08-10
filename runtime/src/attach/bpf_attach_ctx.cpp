/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "attach_private_data.hpp"
#include "base_attach_impl.hpp"
#include "bpftime_shm.hpp"
#include "handler/link_handler.hpp"
#include "handler/prog_handler.hpp"
#include <string>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <map>
#include <memory>
#include <syscall_table.hpp>
#include <bpf_attach_ctx.hpp>
#include <bpftime_shm_internal.hpp>
#include <bpftime_prog.hpp>
#include "bpftime_config.hpp"
#include <spdlog/spdlog.h>
#include <handler/perf_event_handler.hpp>
#include <bpftime_helper_group.hpp>
#include <handler/handler_manager.hpp>
#include <tuple>
#include <utility>
#include <variant>
#include <sys/resource.h>
extern "C" uint64_t bpftime_set_retval(uint64_t value);
namespace bpftime
{

static int load_prog_and_helpers(bpftime_prog *prog, const agent_config &config)
{
	#if __linux__
	if (config.enable_kernel_helper_group) {
		bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog);
	}
	#endif
	if (config.enable_ufunc_helper_group) {
		bpftime_helper_group::get_ufunc_helper_group()
			.add_helper_group_to_prog(prog);
	}
	if (config.enable_shm_maps_helper_group) {
		bpftime_helper_group::get_shm_maps_helper_group()
			.add_helper_group_to_prog(prog);
	}
	return prog->bpftime_prog_load(config.jit_enabled);
}

int bpf_attach_ctx::init_attach_ctx_from_handlers(const agent_config &config)
{
	const handler_manager *manager =
		shm_holder.global_shared_memory.get_manager();
	if (!manager) {
		return -1;
	}
	return init_attach_ctx_from_handlers(manager, config);
}

// create a attach context and progs from handlers
int bpf_attach_ctx::init_attach_ctx_from_handlers(
	const handler_manager *manager, const agent_config &config)
{
	for (int i = 0; i < (int)manager->size(); i++) {
		if (manager->is_allocated(i)) {
			std::set<int> stk;
			if (int err = instantiate_handler_at(manager, i, stk,
							     config);
			    err < 0) {
				SPDLOG_INFO("Failed to instantiate handler {}",
					     i);
				// Unable to instantiate handler may not be an error.
				// We can continue trying to instantiate other handlers.
			}
		}
	}
	return 0;
}

bpf_attach_ctx::~bpf_attach_ctx()
{
	SPDLOG_DEBUG("Destructor: bpf_attach_ctx");
}

// create a probe context
bpf_attach_ctx::bpf_attach_ctx(void)

{
	current_id = CURRENT_ID_OFFSET;
}

int bpf_attach_ctx::instantiate_handler_at(const handler_manager *manager,
					   int id, std::set<int> &stk,
					   const agent_config &config)
{
	SPDLOG_DEBUG("Instantiating handler at {}", id);
	if (instantiated_handlers.contains(id)) {
		SPDLOG_DEBUG("Handler {} already instantiated", id);
		return 0;
	}
	if (stk.contains(id)) {
		SPDLOG_CRITICAL("Loop detected when instantiating handler {}",
				id);
		return -1;
	}
	stk.insert(id);
	auto &handler = manager->get_handler(id);
	if (std::holds_alternative<bpf_prog_handler>(handler)) {
		if (int err = instantiate_prog_handler_at(
			    id, std::get<bpf_prog_handler>(handler), config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate bpf prog handler {}: {}",
				id, err);
			return err;
		}
	} else if (std::holds_alternative<bpf_perf_event_handler>(handler)) {
		if (int err = instantiate_perf_event_handler_at(
			    id, std::get<bpf_perf_event_handler>(handler));
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate bpf perf event handler {}: {}",
				id, err);
			return err;
		}
	} else if (std::holds_alternative<bpf_link_handler>(handler)) {
		auto &link_handler = std::get<bpf_link_handler>(handler);
		if (int err = instantiate_handler_at(
			    manager, link_handler.prog_id, stk, config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate prog handler {} when instantiating link handler {}: {}",
				link_handler.prog_id, id, err);
			return err;
		}
		if (int err = instantiate_handler_at(
			    manager, link_handler.attach_target_id, stk,
			    config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate perf event handler {} when instantiating link handler {}: {}",
				link_handler.attach_target_id, id, err);
			return err;
		}
		if (int err = instantiate_bpf_link_handler_at(id, link_handler);
		    err < 0) {
			SPDLOG_DEBUG(
				"Unable to instantiate bpf link handler {}: {}",
				id, err);
			return err;
		}
	} else {
		SPDLOG_DEBUG("Instantiating type {}", handler.index());
	}
	stk.erase(id);
	instantiated_handlers.insert(id);
	SPDLOG_DEBUG("Instantiating done: {}", id);
	return 0;
}

void bpf_attach_ctx::register_attach_impl(
	std::initializer_list<int> &&attach_types,
	std::unique_ptr<attach::base_attach_impl> &&impl,
	std::function<std::unique_ptr<attach::attach_private_data>(
		const std::string_view &, int &)>
		private_data_creator)
{
	impl->register_custom_helpers([&](unsigned int idx, const char *name,
					  void *func) -> int {
		SPDLOG_INFO("Register attach-impl defined helper {}, index {}",
			    name, idx);
		this->helpers[idx] = bpftime_helper_info{ .index = idx,
							  .name = name,
							  .fn = func };
		return 0;
	});
	auto *impl_ptr = impl.get();
	attach_impl_holders.emplace_back(std::move(impl));
	for (auto ty : attach_types) {
		SPDLOG_DEBUG("Register attach type {} with attach impl {}", ty,
			     typeid(impl_ptr).name());
		attach_impls[ty] =
			std::make_pair(impl_ptr, private_data_creator);
	}
}
int bpf_attach_ctx::instantiate_prog_handler_at(int id,
						const bpf_prog_handler &handler,
						const agent_config &config)
{
	const ebpf_inst *insns = handler.insns.data();
	size_t cnt = handler.insns.size();
	const char *name = handler.name.c_str();
	instantiated_progs[id] =
		std::make_unique<bpftime_prog>(insns, cnt, name);
	bpftime_prog *prog = instantiated_progs[id].get();
	if (int err = load_prog_and_helpers(prog, config); err < 0) {
		SPDLOG_ERROR(
			"Failed to load program helpers for prog handler {}: {}",
			id, err);
		return err;
	}
	for (const auto &item : helpers) {
		prog->bpftime_prog_register_raw_helper(item.second);
	}
	return 0;
}
int bpf_attach_ctx::instantiate_bpf_link_handler_at(
	int id, const bpf_link_handler &handler)
{
	SPDLOG_DEBUG(
		"Instantiating link handler: prog {} -> perf event {}, cookie {}",
		handler.prog_id, handler.attach_target_id,
		handler.attach_cookie.value_or(0));
	auto &[priv_data, attach_type] =
		instantiated_perf_events[handler.attach_target_id];
	attach::base_attach_impl *attach_impl;
	// Find what kind of attach type it is
	if (auto itr = attach_impls.find(attach_type);
	    itr != attach_impls.end()) {
		attach_impl = itr->second.first;
	} else {
		SPDLOG_ERROR("Attach type {} is not registered", attach_type);
		return -ENOTSUP;
	}
	auto prog = instantiated_progs.at(handler.prog_id).get();
	auto cookie = handler.attach_cookie;
	int attach_id = attach_impl->create_attach_with_ebpf_callback(
		[=](void *mem, size_t mem_size, uint64_t *ret) -> int {
			current_thread_bpf_cookie = cookie;
			int err = prog->bpftime_prog_exec((void *)mem, mem_size,
							  ret);
			return err;
		},
		*priv_data, attach_type);
	if (attach_id < 0) {
		// Since the agent might be attach to a unrelated process
		// Using LD_PRELOAD, it's not an error here.
		SPDLOG_DEBUG("Unable to instantiate bpf link handler {}: {}",
			     id, attach_id);
		return attach_id;
	}
	instantiated_attach_links[id] = std::make_pair(attach_id, attach_impl);
	return 0;
}
int bpf_attach_ctx::instantiate_perf_event_handler_at(
	int id, const bpf_perf_event_handler &perf_handler)
{
	SPDLOG_DEBUG("Instantiating perf event handler at {}, type {}", id,
		     (int)perf_handler.type);
	if (perf_handler.type == (int)bpf_event_type::PERF_TYPE_SOFTWARE) {
		SPDLOG_DEBUG(
			"Detected software perf event at {}, nothing need to do",
			id);
		return 0;
	}
	std::unique_ptr<attach::attach_private_data> priv_data;

	auto itr = attach_impls.find((int)perf_handler.type);
	if (itr == attach_impls.end()) {
		SPDLOG_ERROR(
			"Unable to lookup attach implementation of attach type {}",
			(int)perf_handler.type);
		return -ENOENT;
	}
	auto &[attach_impl, private_data_gen] = itr->second;
	if (perf_handler.type ==
		    (int)bpf_event_type::BPF_TYPE_UPROBE_OVERRIDE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_UPROBE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_URETPROBE ||
	    perf_handler.type == (int)bpf_event_type::BPF_TYPE_UREPLACE) {
		auto &uprobe_data =
			std::get<uprobe_perf_event_data>(perf_handler.data);
		std::string arg_str;
		arg_str += uprobe_data._module_name;
		arg_str += ':';
		arg_str += std::to_string(uprobe_data.offset);
		int err = 0;
		priv_data = private_data_gen(arg_str, err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of uprobe perf handler {}, arg_str `{}`: {}",
				id, arg_str, err);
			return err;
		}
	} else if (perf_handler.type ==
		   (int)bpf_event_type::PERF_TYPE_TRACEPOINT) {
		auto &tracepoint_data =
			std::get<tracepoint_perf_event_data>(perf_handler.data);
		int err = 0;
		priv_data = private_data_gen(
			std::to_string(tracepoint_data.tracepoint_id), err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of tracepoint perf handler {}, tp_id `{}`: {}",
				id, tracepoint_data.tracepoint_id, err);
			return err;
		}
	} else {
		auto &custom_data =
			std::get<custom_perf_event_data>(perf_handler.data);
		int err = 0;
		priv_data = private_data_gen(
			std::string(custom_data.attach_argument), err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Unable to parse private data of attach type {}, err={}, raw string={}",
				perf_handler.type, err,
				custom_data.attach_argument);
			return err;
		}
	}
	SPDLOG_DEBUG("Instantiated perf event handler {}", id);
	instantiated_perf_events[id] =
		std::make_pair(std::move(priv_data), (int)perf_handler.type);

	return 0;
}
int bpf_attach_ctx::destroy_instantiated_attach_link(int link_id)
{
	SPDLOG_DEBUG("Destroy attach link {}", link_id);
	if (auto itr = instantiated_attach_links.find(link_id);
	    itr != instantiated_attach_links.end()) {
		auto [attach_id, impl] = itr->second;
		if (int err = impl->detach_by_id(attach_id); err < 0) {
			SPDLOG_ERROR(
				"Failed to detach attach link id {}, attach-specified id {}: {}",
				link_id, attach_id, err);
			return err;
		}
		instantiated_attach_links.erase(itr);
		return 0;
	} else {
		SPDLOG_ERROR("Unable to find instantiated attach link id {}",
			     link_id);
		return -ENOENT;
	}
}
int bpf_attach_ctx::destroy_all_attach_links()
{
	// Avoid modifying along with iterating..
	std::vector<int> to_detach;
	for (const auto &[k, _] : instantiated_attach_links)
		to_detach.push_back(k);
	for (auto k : to_detach) {
		SPDLOG_DEBUG("Destrying attach link {}", k);
		if (int err = destroy_instantiated_attach_link(k); err < 0) {
			SPDLOG_ERROR("Unable to destroy attach link {}: {}", k,
				     err);
			return err;
		}
	}
	return 0;
}
} // namespace bpftime
