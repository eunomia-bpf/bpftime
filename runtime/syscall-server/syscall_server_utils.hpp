/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _SYSCALL_SERVER_UTILS_HPP
#define _SYSCALL_SERVER_UTILS_HPP

#include "syscall_context.hpp"
#include <optional>
#include <filesystem>

int determine_uprobe_perf_type();
int determine_uprobe_retprobe_bit();
void start_up();
std::optional<std::unique_ptr<mocked_file_provider> >
create_mocked_file_based_on_full_path(const std::filesystem::path &path);
std::optional<std::filesystem::path>
resolve_filename_and_fd_to_full_path(int fd, const char *file);
#define PERF_UPROBE_REF_CTR_OFFSET_BITS 32
#define PERF_UPROBE_REF_CTR_OFFSET_SHIFT 32

#endif
