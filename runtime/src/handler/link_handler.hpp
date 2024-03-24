/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LINK_HANDLER_HPP
#define _LINK_HANDLER_HPP

#include "handler/perf_event_handler.hpp"
#include "handler/prog_handler.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <cstdint>
#include <bpftime_shm.hpp>
#include <optional>
#include <variant>
#include <vector>

namespace bpftime
{
constexpr static int BPF_PERF_EVENT = 41;
constexpr static int BPF_TRACE_UPROBE_MULTI = 48;
constexpr static int BPF_F_UPROBE_MULTI_RETURN = 1;
struct perf_event_link_data {
	std::optional<uint64_t> attach_cookie;
};

struct uprobe_multi_entry {
	uint64_t offset;
	uint64_t ref_ctr_offset;
	std::optional<uint64_t> cookie;
};
using uprobe_multi_entry_allocator =
	boost::interprocess::allocator<uprobe_multi_entry,
				       managed_shared_memory::segment_manager>;
using uprobe_multi_entry_vector =
	boost::interprocess::vector<uprobe_multi_entry,
				    uprobe_multi_entry_allocator>;
struct uprobe_multi_link_data {
	boost_shm_string path;
	uprobe_multi_entry_vector entries;
	uint64_t flags;
	int pid;
};

using bpf_link_data =
	std::variant<perf_event_link_data, uprobe_multi_link_data>;

// handle the bpf link fd
struct bpf_link_handler {
	struct bpf_link_create_args args;
	int prog_id;
	int target_id;
	int link_attach_type;
	bpf_link_data data;
	// Create a customized bpf_link
	bpf_link_handler(struct bpf_link_create_args args,
			 managed_shared_memory &mem);
	// Create a bpf_link of type BPF_PERF_EVENT, with provided prog_id and
	// attach_target_id
	bpf_link_handler(int prog_id, int attach_target_id,
			 managed_shared_memory &mem);
	// Create a bpf_link of type BPF_PERF_EVENT, with provided prog_id and
	// attach_target_id, and cookie
	bpf_link_handler(int prog_id, int attach_target_id,
			 std::optional<uint64_t> cookie,
			 managed_shared_memory &mem);
};
} // namespace bpftime

#endif
