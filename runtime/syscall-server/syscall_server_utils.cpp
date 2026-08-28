/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_config.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_shm_internal.hpp"
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
#include "cuda.h"
#endif
#include "syscall_context.hpp"
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <signal.h>
#include <spdlog/spdlog.h>
#include <bpftime_shm.hpp>
#include <string>
#include <system_error>
#include <unistd.h>
#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#include <iomanip>
#include <sstream>
#endif
namespace bpftime
{
static std::once_flag g_startup_once;
static std::exception_ptr g_startup_exception;
static std::string g_syscall_server_shm_name;
static pid_t g_syscall_server_shm_owner_pid = -1;
static bool g_syscall_server_owns_shm = false;
using namespace bpftime;
// Why not use string_view? because parse_uint_from_file requires a c-string
static const std::string UPROBE_TYPE_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/type";
static const std::string URETPROBE_BIT_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/format/retprobe";
static const std::string KPROBE_TYPE_FILE_NAME =
	"/sys/bus/event_source/devices/kprobe/type";
static const std::string KRETPROBE_BIT_FILE_NAME =
	"/sys/bus/event_source/devices/kprobe/format/retprobe";

static bool pid_is_alive(int pid)
{
	return kill(pid, 0) == 0 || errno == EPERM;
}

static void remove_syscall_server_global_shm() noexcept
{
	if (g_syscall_server_shm_name.empty()) {
		return;
	}
	if (getpid() != g_syscall_server_shm_owner_pid) {
		try {
			shm_holder.global_shared_memory
				.remove_pid_from_alive_syscall_server_set(
					getpid());
		} catch (...) {
		}
		return;
	}
	if (!g_syscall_server_owns_shm) {
		try {
			shm_holder.global_shared_memory
				.remove_pid_from_alive_syscall_server_set(
					getpid());
		} catch (...) {
		}
		return;
	}
	shm_lifecycle_lock lifecycle_lock(g_syscall_server_shm_name.c_str());
	try {
		shm_holder.global_shared_memory
			.remove_pid_from_alive_syscall_server_set(getpid());

		bool has_alive_server = false;
		bool server_snapshot_ok =
			shm_holder.global_shared_memory
				.iterate_all_pids_in_alive_syscall_server_set(
					[&](int pid) {
						if (has_alive_server) {
							return;
						}
						if (pid_is_alive(pid)) {
							has_alive_server = true;
						}
					});
		if (!server_snapshot_ok) {
			return;
		}
		if (has_alive_server) {
			return;
		}

		bool has_alive_agent = false;
		bool agent_snapshot_ok =
			shm_holder.global_shared_memory
				.iterate_all_pids_in_alive_agent_set([&](int pid) {
					if (has_alive_agent) {
						return;
					}
					if (pid_is_alive(pid)) {
						has_alive_agent = true;
					}
				});
		if (!agent_snapshot_ok) {
			return;
		}
		if (has_alive_agent) {
			return;
		}
		boost::interprocess::shared_memory_object::remove(
			g_syscall_server_shm_name.c_str());
	} catch (...) {
	}
}

static bool initialize_global_shm_with_ownership()
{
	try {
		bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
		return true;
	} catch (const boost::interprocess::interprocess_exception &error) {
		if (error.get_error_code() !=
		    boost::interprocess::already_exists_error) {
			boost::interprocess::shared_memory_object::remove(
				g_syscall_server_shm_name.c_str());
			throw;
		}
	} catch (...) {
		boost::interprocess::shared_memory_object::remove(
			g_syscall_server_shm_name.c_str());
		throw;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_OR_OPEN);
	return false;
}

void start_up(syscall_context &ctx)
{
	std::call_once(g_startup_once, [&ctx]() {
		try {
			SPDLOG_INFO("Starting syscall server..");
			auto runtime_config = construct_runtime_config_from_env();
			SPDLOG_INFO("Initialize syscall server");

			g_syscall_server_shm_name = get_global_shm_name();
			g_syscall_server_shm_owner_pid = getpid();
			shm_lifecycle_lock lifecycle_lock(
				g_syscall_server_shm_name.c_str());
			g_syscall_server_owns_shm =
				initialize_global_shm_with_ownership();
			std::atexit(remove_syscall_server_global_shm);
			if (!shm_holder.global_shared_memory
				     .add_pid_into_alive_syscall_server_set(
					     getpid())) {
				SPDLOG_WARN(
					"Unable to record alive syscall-server pid; disabling automatic shm removal");
				g_syscall_server_owns_shm = false;
			}
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
			ctx.initialize_cuda();
#endif
			shm_holder.global_shared_memory.begin_new_session();
			shm_holder.global_shared_memory.set_mock_setter(
				[&](bool flg) {
					ctx.enable_mock_after_initialized.store(
						flg, std::memory_order_relaxed);
					SPDLOG_INFO(
						"syscall server: Set enable_mock_after_initialized to {}",
						flg);
				});
#ifdef ENABLE_BPFTIME_VERIFIER
			std::vector<int32_t> helper_ids;
			std::map<int32_t,
				 bpftime::verifier::BpftimeHelperProrotype>
				non_kernel_helpers;
			if (runtime_config.enable_kernel_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_kernel_utils_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
			}
			if (runtime_config.enable_shm_maps_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_shm_maps_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
			}
			if (runtime_config.enable_ufunc_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_shm_maps_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
				// non_kernel_helpers =
				for (const auto &[k, v] :
				     get_ufunc_helper_protos()) {
					non_kernel_helpers[k] = v;
				}
			}
			verifier::set_available_helpers(helper_ids);
			SPDLOG_INFO("Enabling {} helpers", helper_ids.size());
			verifier::set_non_kernel_helpers(non_kernel_helpers);
#endif
			bpftime_set_runtime_config(std::move(runtime_config));
			// Set a variable to indicate the program that it's
			// controlled by bpftime
			setenv("BPFTIME_USED", "1", 0);
			SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
			SPDLOG_INFO("bpftime-syscall-server started");
		} catch (...) {
			g_startup_exception = std::current_exception();
		}
	});
	if (g_startup_exception)
		std::rethrow_exception(g_startup_exception);
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
int determine_kprobe_perf_type()
{
	if (!std::filesystem::exists(KPROBE_TYPE_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked kprobe type value {} for file {}",
			     MOCKED_KPROBE_TYPE_VALUE, KPROBE_TYPE_FILE_NAME);
		return MOCKED_KPROBE_TYPE_VALUE;
	}
	return parse_uint_from_file(KPROBE_TYPE_FILE_NAME.c_str(), "%d\n");
}
int determine_kprobe_retprobe_bit()
{
	if (!std::filesystem::exists(KRETPROBE_BIT_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uretprobe bit value {} for file {}",
			     MOCKED_KRETPROBE_BIT, KRETPROBE_BIT_FILE_NAME);
		return MOCKED_KRETPROBE_BIT;
	}
	return parse_uint_from_file(KRETPROBE_BIT_FILE_NAME.c_str(),
				    "config:%d\n");
}
std::optional<std::unique_ptr<mocked_file_provider>>
create_mocked_file_based_on_full_path(const std::filesystem::path &path)
{
	auto path_text = path.string();
	const bool is_tracepoint_id =
		(path_text.starts_with("/sys/kernel/debug/tracing/events/") ||
		 path_text.starts_with("/sys/kernel/tracing/events/")) &&
		path_text.ends_with("/id");
	if (is_tracepoint_id) {
		uint32_t hash = 2166136261U;
		for (unsigned char ch : path_text) {
			hash ^= ch;
			hash *= 16777619U;
		}
		return std::make_unique<mocked_file_provider>(
			std::to_string(1024U + hash % 100000000U) + "\n");
	}
	if (path == UPROBE_TYPE_FILE_NAME) {
		SPDLOG_DEBUG("{} is uprobe type file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			std::to_string(MOCKED_UPROBE_TYPE_VALUE));
	} else if (path == URETPROBE_BIT_FILE_NAME) {
		SPDLOG_DEBUG("{} is uretprobe bit file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			"config:" + std::to_string(MOCKED_URETPROBE_BIT));
	} else if (path == KPROBE_TYPE_FILE_NAME) {
		SPDLOG_DEBUG("{} is kprobe type file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			std::to_string(MOCKED_KPROBE_TYPE_VALUE));
	} else if (path == KRETPROBE_BIT_FILE_NAME) {
		SPDLOG_DEBUG("{} is kretprobe bit file", path.c_str());
		return std::make_unique<mocked_file_provider>(
			"config:" + std::to_string(MOCKED_KRETPROBE_BIT));
	} else {
		SPDLOG_DEBUG("Unmocked file path: {}", path.c_str());
		return {};
	}
}

std::optional<std::filesystem::path>
resolve_filename_and_fd_to_full_path(int fd, const char *file)
{
	if (file == nullptr) {
		return {};
	}
	if (file[0] == '/') {
		return std::filesystem::path(file);
	}
	if (fd == AT_FDCWD) {
		return std::filesystem::path(file);
	}
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

} // namespace bpftime
