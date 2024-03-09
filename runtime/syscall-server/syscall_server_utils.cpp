/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_shm_internal.hpp"
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <bpftime_shm.hpp>
#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#include <iomanip>
#include <sstream>
#endif
static bool already_setup = false;
static bool disable_mock = true;
using namespace bpftime;

void start_up()
{
	if (already_setup)
		return;
	already_setup = true;
	SPDLOG_INFO("Initialize syscall server");
	spdlog::cfg::load_env_levels();
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	const auto agent_config = get_agent_config_from_env();
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

const char *expand_tilde(const char *path) {
    if (path[0] == '~' && (path[1] == '/' || path[1] == '\0')) {
        const char *home_dir = getenv("HOME");
        if (home_dir != NULL) {
            // Allocate memory for the expanded path
            size_t len = strlen(home_dir) + strlen(path) - 1;  // -1 to exclude ~
            char *expanded_path = (char *)malloc(len + 1);     // +1 for the null terminator

            if (expanded_path != NULL) {
                // Construct the expanded path
                strcpy(expanded_path, home_dir);
                strcat(expanded_path, path + 1);  // Skip the ~ at the beginning

                return expanded_path;
            } else {
                fprintf(stderr, "Failed to allocate memory\n");
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "HOME environment variable is not set\n");
            exit(EXIT_FAILURE);
        }
    } else {
        // If the path doesn't start with ~, return it unchanged
        return strdup(path);
    }
}

int determine_uprobe_perf_type()
{
	const char *file = "/sys/bus/event_source/devices/uprobe/type";
	FILE *f;
	
	f = fopen(file, "re");
	if(!f){
		const char *file_with_tilde = "~/.bpftime/event_source/devices/uprobe/type";
		const char *file = expand_tilde(file_with_tilde);
		
		return parse_uint_from_file(file, "%d\n");
	}

	return parse_uint_from_file(file, "%d\n");
}

int determine_uprobe_retprobe_bit()
{
	const char *file =
		"/sys/bus/event_source/devices/uprobe/format/retprobe";
	FILE *f;

	f = fopen(file, "re");
	if(!f){
		const char *file_with_tilde = "~/.bpftime/event_source/devices/uprobe/format/retprobe";
		const char *file = expand_tilde(file_with_tilde);
		
		return parse_uint_from_file(file, "config:%d\n");
	}

	return parse_uint_from_file(file, "config:%d\n");
}
