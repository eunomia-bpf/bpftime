/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LINK_HANDLER_HPP
#define _LINK_HANDLER_HPP
#include <cstdint>
namespace bpftime
{
// handle the bpf link fd
struct bpf_link_handler {
	uint32_t prog_fd, target_fd;
};
} // namespace bpftime

#endif
