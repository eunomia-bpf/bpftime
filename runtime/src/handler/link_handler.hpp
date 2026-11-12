/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LINK_HANDLER_HPP
#define _LINK_HANDLER_HPP

#include <cstdint>
#include <bpftime_shm.hpp>
#include <optional>

namespace bpftime
{
// handle the bpf link fd
struct bpf_link_handler {
	struct bpf_link_create_args args;
	int prog_id;
	int attach_target_id;
	std::optional<uint64_t> attach_cookie;
	bpf_link_handler(struct bpf_link_create_args args)
		: args(args), prog_id(args.prog_fd),
		  attach_target_id(args.target_fd)
	{
		// Only perf-event links store the cookie in perf_event.bpf_cookie.
		// Other attach types keep it in a different union member
		// (tracing.cookie, kprobe_multi.cookies, ...), so leave
		// attach_cookie disengaged for them instead of reading the wrong
		// field.
		if (args.attach_type == BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE)
			attach_cookie = args.perf_event.bpf_cookie;
	}
	bpf_link_handler(int prog_id, int attach_target_id)
		: prog_id(prog_id), attach_target_id(attach_target_id)
	{
	}
	bpf_link_handler(int prog_id, int attach_target_id,
			 std::optional<uint64_t> cookie)
		: prog_id(prog_id), attach_target_id(attach_target_id),
		  attach_cookie(cookie)
	{
	}
};
} // namespace bpftime

#endif
