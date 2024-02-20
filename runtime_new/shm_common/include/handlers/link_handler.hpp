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
namespace shm_common
{
// handle the bpf link fd
struct bpf_link_handler {
	int prog_id;
	int attach_target_id;
};
} // namespace shm_common
} // namespace bpftime

#endif
