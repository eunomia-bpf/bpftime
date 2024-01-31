/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _LINK_HANDLER_HPP
#define _LINK_HANDLER_HPP

#include <cstdint>
#include <bpftime_shm.hpp>

namespace bpftime
{
// handle the bpf link fd
struct bpf_link_handler {
	struct bpf_link_create_args args;
};
} // namespace bpftime

#endif
