/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_TRAP_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_TRAP_ATTACH_PRIVATE_DATA_HPP
#include "attach_private_data.hpp"
#include <cstdint>
#include <string>
namespace bpftime
{
namespace attach
{
namespace trap
{
// Private data for the trap uprobe backend. Accepts the same string format
// as the frida backend: either a decimal function address, or
// `MODULE:OFFSET` where MODULE is a path (empty means the main executable)
// and OFFSET is the file offset of the function inside that module.
struct trap_attach_private_data final : public attach_private_data {
	uint64_t addr = 0;
	std::string module_name;
	int initialize_from_string(const std::string_view &sv) override;
	std::string to_string() const override;
};
} // namespace trap
} // namespace attach
} // namespace bpftime
#endif
