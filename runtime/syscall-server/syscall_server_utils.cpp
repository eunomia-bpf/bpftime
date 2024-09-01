/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm_internal.hpp"
#include "syscall_context.hpp"
#include <filesystem>
#include <memory>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include "bpftime_logger.hpp"
#include <bpftime_shm.hpp>
#include <string>
#include <system_error>
#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#include <iomanip>
#include <sstream>
#endif
static bool already_setup = false;
static bool disable_mock = true;
using namespace bpftime;
// Why not use string_view? because parse_uint_from_file requires a c-string
static const std::string UPROBE_TYPE_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/type";
static const std::string URETPROBE_BIT_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/format/retprobe";

void start_up()
{
	if (already_setup)
		return;
	already_setup = true;
	const auto agent_config = get_agent_config_from_env();
	bpftime_set_logger(agent_config.logger_output_path);
	SPDLOG_INFO("Initialize syscall server");

	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	bpftime_set_agent_config(agent_config);
#ifdef ENABLE_BPFTIME_VERIFIER
	std::vector<int32_t> helper_ids;
	std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype>
		non_kernel_helpers;
	if (agent_config.enable_kernel_helper_group) {
		for (auto x :
		     bpftime_helper_group::get_kernel_utils_helper_group()
			     .get_helper_ids()) {
			helper_ids.push_back(x);
		}
	}
	if (agent_config.enable_shm_maps_helper_group) {
		for (auto x : bpftime_helper_group::get_shm_maps_helper_group()
				      .get_helper_ids()) {
			helper_ids.push_back(x);
		}
	}
	if (agent_config.enable_ufunc_helper_group) {
		for (auto x : bpftime_helper_group::get_shm_maps_helper_group()
				      .get_helper_ids()) {
			helper_ids.push_back(x);
		}
		// non_kernel_helpers =
		for (const auto &[k, v] : get_ufunc_helper_protos()) {
			non_kernel_helpers[k] = v;
		}
	}
	verifier::set_available_helpers(helper_ids);
	SPDLOG_INFO("Enabling {} helpers", helper_ids.size());
	verifier::set_non_kernel_helpers(non_kernel_helpers);
#endif
	// Set a variable to indicate the program that it's controlled by
	// bpftime
	setenv("BPFTIME_USED", "1", 0);
	SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
	SPDLOG_INFO("bpftime-syscall-server started");
}

/*
 * this function is expected to parse integer in the range of [0, 2^31-1] from
 * given file using scanf format string fmt. If actual parsed value is
 * negative, the result might be indistinguishable from error
 */
static int parse_uint_from_file(const char *file, const char *fmt)
{
	int err, ret;
	FILE *f;

	f = fopen(file, "re");
	if (!f) {
		err = -errno;
		SPDLOG_ERROR("Failed to open {}: {}", file, err);
		return err;
	}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	err = fscanf(f, fmt, &ret);
#pragma GCC diagnostic pop
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		SPDLOG_ERROR("Failed to parse {}: {}", file, err);
		fclose(f);
		return err;
	}
	fclose(f);
	return ret;
}

int determine_uprobe_perf_type()
{
	if (!std::filesystem::exists(UPROBE_TYPE_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uporbe type value {} for file {}",
			     MOCKED_UPROBE_TYPE_VALUE, UPROBE_TYPE_FILE_NAME);
		return MOCKED_UPROBE_TYPE_VALUE;
	}
	return parse_uint_from_file(UPROBE_TYPE_FILE_NAME.c_str(), "%d\n");
}

int determine_uprobe_retprobe_bit()
{
	if (!std::filesystem::exists(URETPROBE_BIT_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uretprobe bit value {} for file {}",
			     MOCKED_URETPROBE_BIT, URETPROBE_BIT_FILE_NAME);
		return MOCKED_URETPROBE_BIT;
	}
	return parse_uint_from_file(URETPROBE_BIT_FILE_NAME.c_str(),
				    "config:%d\n");
}

std::optional<std::unique_ptr<mocked_file_provider> >
create_mocked_file_based_on_full_path(const std::filesystem::path &path)
{
	if (path == UPROBE_TYPE_FILE_NAME) {
		SPDLOG_DEBUG("{} is uprobe type file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			std::to_string(MOCKED_UPROBE_TYPE_VALUE));
	} else if (path == URETPROBE_BIT_FILE_NAME) {
		SPDLOG_DEBUG("{} is uretprobe bit file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			"config:" + std::to_string(MOCKED_URETPROBE_BIT));
	} else {
		SPDLOG_DEBUG("Unmocked file path: {}", path.c_str());
		return {};
	}
}

std::optional<std::filesystem::path>
resolve_filename_and_fd_to_full_path(int fd, const char *file)
{
	std::error_code ec;
	auto dir_path = std::filesystem::read_symlink(
		"/proc/self/fd/" + std::to_string(fd), ec);
	if (dir_path.empty()) {
		SPDLOG_ERROR("Unable to read exact path of fd {}, error={}: ",
			     fd, ec.value(), ec.message());
		return {};
	}
	if (!std::filesystem::is_directory(dir_path)) {
		SPDLOG_ERROR("fd {}, referring {}, is not a directory", fd,
			     dir_path.c_str());
		return {};
	}
	return dir_path / file;
}
