/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Symbol and module resolution helpers that do not depend on frida-gum.
#ifndef _BPFTIME_TRAP_ATTACH_UTILS_HPP
#define _BPFTIME_TRAP_ATTACH_UTILS_HPP
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
namespace bpftime
{
namespace attach
{
namespace trap
{
// Address at which file offset 0 of `module_name` is mapped. An empty name
// selects the main executable. Returns nullptr when the module is not
// loaded.
void *get_module_base_addr(const char *module_name);
// Translate a file offset inside a module into a runtime address.
void *
resolve_function_addr_by_module_offset(const std::string_view &module_name,
				       uintptr_t func_offset);
// Resolve /proc/<pid>/map_files/<start>-<end> to the underlying mapped path.
// Other module names are returned unchanged.
std::optional<std::string>
resolve_mapped_module_path(const std::string_view &module_name);
// Look a function up by symbol name across every loaded module. Both the
// dynamic and the regular symbol table are consulted, so non exported
// functions of unstripped binaries are found as well.
void *find_function_addr_by_name(const char *name);
// Find an exported symbol of a specific module.
void *find_module_export_by_name(const char *module_name,
				 const char *symbol_name);
} // namespace trap
} // namespace attach
} // namespace bpftime
#endif
