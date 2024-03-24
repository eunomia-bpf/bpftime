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
#include <optional>
#include <ostream>
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
#include "spdlog/fmt/ostr.h"

extern "C" uint64_t bpftime_set_retval(uint64_t value);
namespace bpftime
{

static int load_prog_and_helpers(bpftime_prog *prog, const agent_config &config)
{
	if (config.enable_kernel_helper_group) {
		bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog);
	}
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
				SPDLOG_ERROR("Failed to instantiate handler {}",
					     i);
				return err;
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
	// Register handlers for different link attach type
	link_attach_handlers[BPF_PERF_EVENT] =
		[this](const bpf_link_handler &link, int id,
		       const handler_manager *manager)
		-> std::optional<instantiated_link_record> {
		// Need to instantiate the corresponding perf event first
		if (int err = instantiate_perf_event_handler_at(
			    link.target_id,
			    (std::get<bpf_perf_event_handler>(
				    manager->get_handler(link.target_id))));
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate perf event handler {} when instantiating link handler {}: {}",
				link.target_id, id, err);
			return {};
		}
		// For normal perf events, we instantiate by calling
		// instantiate_perf_event_bpf_link_handler_at
		return instantiate_perf_event_bpf_link_handler_at(id, link,
								  manager);
	};
	link_attach_handlers[BPF_TRACE_UPROBE_MULTI] =
		[this](const bpf_link_handler &link, int id,
		       const handler_manager *manager)
		-> std::optional<instantiated_link_record> {
		return this->instantiate_uprobe_multi_handler_at(id, link);
	};
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
		// Instantiate program of the current bpf link
		if (int err = instantiate_handler_at(
			    manager, link_handler.prog_id, stk, config);
		    err < 0) {
			SPDLOG_ERROR(
				"Unable to instantiate prog handler {} when instantiating link handler {}: {}",
				link_handler.prog_id, id, err);
			return err;
		}
		// Call the related handler to handler the instantiation of the
		// link. Handler might instantiate other handlers.
		if (auto itr = link_attach_handlers.find(
			    link_handler.link_attach_type);
		    itr != link_attach_handlers.end()) {
			if (auto ret = itr->second(link_handler, id, manager);
			    ret.has_value()) {
				instantiated_attach_ids[id] = ret.value();
			} else {
				SPDLOG_ERROR(
					"Unable to instantiate link handler {}",
					id);
				return -EINVAL;
			}
		} else {
			SPDLOG_ERROR("Unsupported link attach type: {}",
				     link_handler.link_attach_type);
			return -ENOTSUP;
		}
	} else if (std::holds_alternative<unused_handler>(handler)) {
		SPDLOG_ERROR("Instantiating a unused handler at {}", id);
		return -EINVAL;
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
std::optional<instantiated_link_record>
bpf_attach_ctx::instantiate_uprobe_multi_handler_at(
	int id, const bpf_link_handler &handler)
{
	SPDLOG_DEBUG(
		"Instantiating bpf link handler of type uprobe multi at {}",
		id);
	auto prog = instantiated_progs.at(handler.prog_id).get();
	// Instantiate uprobe multi link handler
	// Such type doesn't have an attach target, so we direct create uprobes
	const auto &link_data = std::get<uprobe_multi_link_data>(handler.data);

	attach::base_attach_impl *attach_impl = nullptr;
	private_data_creator priv_creator;
	int attach_type;
	if (link_data.flags & BPF_F_UPROBE_MULTI_RETURN) {
		// Here we should ensure that attach impl of type uretprobe has
		// been registered
		if (auto itr = attach_impls.find(
			    (int)bpf_event_type::BPF_TYPE_URETPROBE);
		    itr != attach_impls.end()) {
			std::tie(attach_impl, priv_creator) = itr->second;
			attach_type = (int)bpf_event_type::BPF_TYPE_URETPROBE;

		} else {
			SPDLOG_ERROR(
				"Trying to instantiate uprobe multi (exit hook), but uretprobe is not registered");
			return {};
		}
	} else {
		// been registered
		if (auto itr = attach_impls.find(
			    (int)bpf_event_type::BPF_TYPE_UPROBE);
		    itr != attach_impls.end()) {
			std::tie(attach_impl, priv_creator) = itr->second;
			attach_type = (int)bpf_event_type::BPF_TYPE_UPROBE;

		} else {
			SPDLOG_ERROR(
				"Trying to instantiate uprobe multi (entry hook), but uretprobe is not registered");
			return {};
		}
	}
	instantiated_link_record native_ids;
	for (const auto &entry : link_data.entries) {
		std::string arg_str = link_data.path.c_str();
		arg_str += ":";
		arg_str += std::to_string(entry.offset);
		int err = 0;
		auto priv_data = priv_creator(arg_str, err);
		if (err < 0) {
			SPDLOG_ERROR(
				"Failed to create uprobe/uretprobe private data for arg string {}, err={}, uprobe multi link {}",
				arg_str, err, id);
			return {};
		}
		auto cookie = entry.cookie;

		int attach_id = attach_impl->create_attach_with_ebpf_callback(
			[=](void *mem, size_t mem_size, uint64_t *ret) -> int {
				current_thread_bpf_cookie = cookie;
				int err = prog->bpftime_prog_exec(
					(void *)mem, mem_size, ret);
				return err;
			},
			*priv_data, attach_type);
		native_ids.emplace_back(attach_id, attach_impl);
	}
	return native_ids;
}
std::optional<instantiated_link_record>
bpf_attach_ctx::instantiate_perf_event_bpf_link_handler_at(
	int id, const bpf_link_handler &handler, const handler_manager *manager)
{
	SPDLOG_DEBUG(
		"Instantiating bpf link handler of type perf event at id {}, prog {}, target {}",
		id, handler.prog_id, handler.target_id);
	auto prog = instantiated_progs.at(handler.prog_id).get();
	// For perf event link, there should be an instantiated target
	auto &[priv_data, attach_type] =
		instantiated_perf_events[handler.target_id];
	SPDLOG_DEBUG("Attach private data is {}, attach type is ",
		     priv_data->to_string(), attach_type);
	attach::base_attach_impl *attach_impl;
	if (auto itr = attach_impls.find(attach_type);
	    itr != attach_impls.end()) {
		attach_impl = itr->second.first;
	} else {
		SPDLOG_ERROR("Attach type {} is not registered", attach_type);
		return {};
	}
	auto cookie =
		std::get<perf_event_link_data>(handler.data).attach_cookie;
	if (cookie.has_value()) {
		SPDLOG_DEBUG("Attach cookie is {}", cookie.value());
	}
	int attach_id = attach_impl->create_attach_with_ebpf_callback(
		[=](void *mem, size_t mem_size, uint64_t *ret) -> int {
			current_thread_bpf_cookie = cookie;
			int err = prog->bpftime_prog_exec((void *)mem, mem_size,
							  ret);
			return err;
		},
		*priv_data, attach_type);
	if (attach_id < 0) {
		SPDLOG_ERROR(
			"Unable to instantiate bpf link handler {} (sub perf id {}): {}",
			id, handler.target_id, attach_id);
		return {};
	}
	return std::optional<instantiated_link_record>(
		{ std::make_pair(attach_id, attach_impl) });
}

// int bpf_attach_ctx::instantiate_bpf_link_handler_at(
// 	int id, const bpf_link_handler &handler)
// {
// 	SPDLOG_DEBUG(
// 		"Instantiating link handler ({}): prog {} -> perf event count ({}), link_attach_type {}",
// 		id, handler.prog_id, handler.attach_target_ids.size(),
// 		handler.link_attach_type);

// 	auto prog = instantiated_progs.at(handler.prog_id).get();
// 	if (handler.link_attach_type == BPF_PERF_EVENT ||
// 	    handler.link_attach_type == BPF_TRACE_UPROBE_MULTI) {
// 		std::vector<std::pair<int, attach::base_attach_impl *> >
// 			internal_attach_records;
// 		int i = 0;
// 		for (auto target_id : handler.attach_target_ids) {
// 			SPDLOG_DEBUG(
// 				"Handling sub attach target with target id {}, index {}",
// 				target_id, i);
// 			auto &[priv_data, attach_type] =
// 				instantiated_perf_events[target_id];
// 			SPDLOG_DEBUG(
// 				"Attach private data is {}, attach type is ",
// 				priv_data->to_string(), attach_type);
// 			attach::base_attach_impl *attach_impl;
// 			if (auto itr = attach_impls.find(attach_type);
// 			    itr != attach_impls.end()) {
// 				attach_impl = itr->second.first;
// 			} else {
// 				SPDLOG_ERROR("Attach type {} is not registered",
// 					     attach_type);
// 				return -ENOTSUP;
// 			}

// 			std::optional<uint64_t> cookie;
// 			if (std::holds_alternative<perf_event_link_data>(
// 				    handler.data)) {
// 				cookie = std::get<perf_event_link_data>(
// 						 handler.data)
// 						 .attach_cookie;

// 			} else if (std::holds_alternative<uprobe_multi_link_data>(
// 					   handler.data)) {
// 				cookie = std::get<uprobe_multi_link_data>(
// 						 handler.data)
// 						 .entries[i]
// 						 .cookie;
// 			}
// 			if (cookie.has_value()) {
// 				SPDLOG_DEBUG("Attach cookie is {}",
// 					     cookie.value());
// 			}
// 			int attach_id =
// 				attach_impl->create_attach_with_ebpf_callback(
// 					[=](void *mem, size_t mem_size,
// 					    uint64_t *ret) -> int {
// 						current_thread_bpf_cookie =
// 							cookie;
// 						int err =
// 							prog->bpftime_prog_exec(
// 								(void *)mem,
// 								mem_size, ret);
// 						return err;
// 					},
// 					*priv_data, attach_type);
// 			if (attach_id < 0) {
// 				SPDLOG_ERROR(
// 					"Unable to instantiate bpf link handler {} (sub perf id {}): {}",
// 					id, target_id, attach_id);
// 				return attach_id;
// 			}
// 			internal_attach_records.emplace_back(attach_id,
// 							     attach_impl);
// 			i++;
// 		}
// 		instantiated_attach_ids[id] = internal_attach_records;

// 	} else {
// 		SPDLOG_ERROR("We does not support link with attach type {} yet",
// 			     handler.link_attach_type);
// 		return -ENOTSUP;
// 	}
// 	return 0;
// }
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
	if (auto itr = instantiated_attach_ids.find(link_id);
	    itr != instantiated_attach_ids.end()) {
		for (const auto &[attach_id, impl] : itr->second) {
			SPDLOG_DEBUG("Destroy sub attach {}", attach_id);
			if (int err = impl->detach_by_id(attach_id); err < 0) {
				SPDLOG_ERROR(
					"Failed to detach attach link id {}, attach-specified id {}: {}",
					link_id, attach_id, err);
				return err;
			}
		}
		instantiated_attach_ids.erase(itr);
		return 0;
	} else {
		SPDLOG_ERROR("Unable to find instantiated attach link id {}",
			     link_id);
		return -ENOENT;
	}
}
} // namespace bpftime
