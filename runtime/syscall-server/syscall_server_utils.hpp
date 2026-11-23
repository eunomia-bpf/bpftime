/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _SYSCALL_SERVER_UTILS_HPP
#define _SYSCALL_SERVER_UTILS_HPP

#include "syscall_context.hpp"
#include <cstdint>
#include <optional>
#include <filesystem>

namespace bpftime
{
int determine_uprobe_perf_type();
int determine_uprobe_retprobe_bit();
int determine_kprobe_perf_type();
int determine_kprobe_retprobe_bit();

void start_up(syscall_context &ctx);
// Compute the generated content for a virtual synthetic file (rootless
// tracefs event id, event_source type/format, hidden-address kallsyms).
// Returns nullopt when the virtual path must be served by the real file.
std::optional<std::string>
get_synthetic_file_content(const std::filesystem::path &path);
// Resolve a virtual synthetic path to a real host file path the caller can
// open/stat directly. Explicit BPFTIME_KALLSYMS_PATH/BPFTIME_BTF_PATH
// overrides are returned as-is; generated files are materialized once into a
// private per-process temporary directory and cached. Nullopt means the
// path needs no redirect.
std::optional<std::string>
resolve_redirected_path(const std::filesystem::path &full_path);
std::optional<std::filesystem::path>
resolve_filename_and_fd_to_full_path(int fd, const char *file);
constexpr uint64_t PERF_UPROBE_REF_CTR_OFFSET_SHIFT = 32;

} // namespace bpftime

#endif
