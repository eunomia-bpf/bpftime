/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LINK_HANDLER_HPP
#define _LINK_HANDLER_HPP

#include <cstdint>
#include <optional>
namespace bpftime
{
namespace shm_common
{
// Represent a link between program and attach target
struct bpf_link_handler {
	// The program id
	int prog_id;
	// The attach target id
	int attach_target_id;
	// The attach cookie, used to distinguish different attaches by the ebpf program
	std::optional<uint64_t> cookie;
};
} // namespace shm_common
} // namespace bpftime

#endif
