/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "handlers/link_handler.hpp"
#include "handlers/perf_event_handler.hpp"
#include "handlers/prog_handler.hpp"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <handler_manager.hpp>
#include <variant>
#include <algorithm>

using namespace bpftime::shm_common;
handler_manager::handler_manager(managed_shared_memory &mem,
				 size_t max_id_count)
	: handlers(max_id_count,
		   handler_variant_allocator(mem.get_segment_manager()))
{
}

handler_manager::~handler_manager()
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		assert(!is_allocated(i));
	}
}

const handler_variant &handler_manager::get_handler(int id) const
{
	return handlers[id];
}

const handler_variant &handler_manager::operator[](int idx) const
{
	return handlers[idx];
}
std::size_t handler_manager::size() const
{
	return handlers.size();
}

int handler_manager::set_handler(int id, handler_variant &&handler,
				 managed_shared_memory &memory)
{
	if (is_allocated(id)) {
		SPDLOG_ERROR("set_handler failed for fd {} aleady exists", id);
		return -ENOENT;
	}
	destroy_handler_at(id, memory);
	handlers[id] = std::move(handler);
	if (std::holds_alternative<bpf_map_handler>(handlers[id])) {
		std::get<bpf_map_handler>(handlers[id]).map_init(memory);
	}
	return id;
}

bool handler_manager::is_allocated(int id) const
{
	if ((size_t)id >= handlers.size()) {
		SPDLOG_WARN("ID is too large!");
	}
	if (id < 0 || (std::size_t)id >= handlers.size()) {
		return false;
	}
	return !std::holds_alternative<unused_handler>(handlers.at(id));
}

void handler_manager::destroy_handler_at(int id, managed_shared_memory &memory)
{
	if (id < 0 || (std::size_t)id >= handlers.size()) {
		return;
	}
	if (std::holds_alternative<bpf_map_handler>(handlers[id])) {
		std::get<bpf_map_handler>(handlers[id]).map_free(memory);
	} else if (std::holds_alternative<bpf_perf_event_handler>(
			   handlers[id])) {
		// Clean related links
		SPDLOG_DEBUG("Destroying perf event handler {}", fd);
		for (size_t i = 0; i < handlers.size(); i++) {
			auto &handler = handlers[i];
			if (std::holds_alternative<bpf_link_handler>(handler) &&
			    std::get<bpf_link_handler>(handler)
					    .attach_target_id == id) {
				SPDLOG_DEBUG(
					"Remove link {} by removing perf event {}",
					i, fd);
				destroy_handler_at(i, memory);
			}
		}
	} else if (std::holds_alternative<bpf_prog_handler>(handlers[id])) {
		// Destroy links
		SPDLOG_DEBUG("Destroying program handler {}", fd);
		for (size_t i = 0; i < handlers.size(); i++) {
			auto &handler = handlers[i];
			if (std::holds_alternative<bpf_link_handler>(handler) &&
			    std::get<bpf_link_handler>(handler).prog_id == id) {
				SPDLOG_DEBUG(
					"Remove link {} by removing program {}",
					i, fd);
				destroy_handler_at(i, memory);
			}
		}
	}
	handlers[id] = unused_handler();
}

int handler_manager::find_minimal_unused_idx() const
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		if (!is_allocated(i)) {
			return i;
		}
	}
	return -1;
}

void handler_manager::clear_all(managed_shared_memory &memory)
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		if (is_allocated(i)) {
			destroy_handler_at(i, memory);
		}
	}
}
