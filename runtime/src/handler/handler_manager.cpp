/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "handler/link_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "handler/prog_handler.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <handler/handler_manager.hpp>
#include <variant>
#include <algorithm>
#if __APPLE__
#include "spinlock_wrapper.hpp"
#endif
namespace bpftime
{
handler_manager::handler_manager(managed_shared_memory &mem,
				 size_t max_fd_count)
	: handlers(max_fd_count,
		   handler_variant_allocator(mem.get_segment_manager()))
{
}

handler_manager::~handler_manager()
{
	for (std::size_t i = 0; i < handlers.size(); i++) {
		SPDLOG_TRACE(
			"Handler at {} is not destroyed, but handler_manager is being destroyed",
			i);
	}
}

const handler_variant &handler_manager::get_handler(int fd) const
{
	return handlers[fd];
}

const handler_variant &handler_manager::operator[](int idx) const
{
	return handlers[idx];
}
std::size_t handler_manager::size() const
{
	return handlers.size();
}

int handler_manager::set_handler_at_empty_slot(handler_variant &&handler,
					       managed_shared_memory &memory)
{
	int id = find_minimal_unused_idx();
	if (id < 0) {
		SPDLOG_ERROR("Unable to find empty slot");
		return id;
	}
	return set_handler(id, std::move(handler), memory);
}
int handler_manager::set_handler(int fd, handler_variant &&handler,
				 managed_shared_memory &memory)
{
	if (is_allocated(fd)) {
		SPDLOG_ERROR("set_handler failed for fd {} aleady exists", fd);
		return -EEXIST;
	}
	if (std::holds_alternative<unused_handler>(handler)) {
		SPDLOG_ERROR(
			"Unable to set a handler to unused_handler with set_handler, please use clear_id_at");
		return -ENOTSUP;
	}
	SPDLOG_DEBUG("Handler at fd {} set to type {}", fd, handler.index());
	handlers[fd] = std::move(handler);
	if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
		std::get<bpf_map_handler>(handlers[fd]).map_init(memory);
	}
	return fd;
}

bool handler_manager::is_allocated(int fd) const
{
	if (fd < 0 || (std::size_t)fd >= handlers.size()) {
		return false;
	}
	return !std::holds_alternative<unused_handler>(handlers.at(fd));
}

void handler_manager::clear_id_at(int fd, managed_shared_memory &memory)
{
	if (fd < 0 || (std::size_t)fd >= handlers.size()) {
		return;
	}
	if (std::holds_alternative<bpf_map_handler>(handlers[fd])) {
		std::get<bpf_map_handler>(handlers[fd]).map_free(memory);
	} else if (std::holds_alternative<bpf_perf_event_handler>(
			   handlers[fd])) {
		// Clean attached programs..
		SPDLOG_DEBUG("Destroying perf event handler {}", fd);
		for (size_t i = 0; i < handlers.size(); i++) {
			auto &handler = handlers[i];
			if (std::holds_alternative<bpf_link_handler>(handler)) {
				auto &link_handler =
					std::get<bpf_link_handler>(handler);
				if (link_handler.attach_target_id == fd) {
					SPDLOG_DEBUG(
						"Remove link handler with id {}, prog id {}, due to the removal of perf event {}",
						i, link_handler.prog_id, fd);
					clear_id_at(i, memory);
				}
			}
		}
	} else if (std::holds_alternative<bpf_prog_handler>(handlers[fd])) {
		SPDLOG_DEBUG("Destroying prog ehandler {}", fd);
		for (size_t i = 0; i < handlers.size(); i++) {
			auto &handler = handlers[i];
			if (std::holds_alternative<bpf_link_handler>(handler)) {
				auto &link_handler =
					std::get<bpf_link_handler>(handler);
				if (link_handler.prog_id == fd) {
					SPDLOG_DEBUG(
						"Remove link handler with id {}, perf event id {}, due to the removal of perf event {}",
						i,
						link_handler.attach_target_id,
						fd);
					clear_id_at(i, memory);
				}
			}
		}
	}
	handlers[fd] = unused_handler();
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
			clear_id_at(i, memory);
		}
	}
}

} // namespace bpftime
