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

static void process_token(const std::string_view &token, agent_config &config)
{
	if (token == "ffi") {
		spdlog::info("Enabling ffi helper group");
		config.enable_ffi_helper_group = true;
	} else if (token == "kernel") {
		spdlog::info("Enabling kernel helper group");
		config.enable_kernel_helper_group = true;
	} else if (token == "shm_map") {
		spdlog::info("Enabling shm_map helper group");
		config.enable_shm_maps_helper_group = true;
	} else {
		spdlog::warn("Unknown helper group: {}", token);
	}
}

static void process_helper_sv(const std::string_view &str, const char delimiter,
			      agent_config &config)
{
	std::string::size_type start = 0;
	std::string::size_type end = str.find(delimiter);

	while (end != std::string::npos) {
		process_token(str.substr(start, end - start), config);
		start = end + 1;
		end = str.find(delimiter, start);
	}

	// Handle the last token, if any
	if (start < str.size()) {
		process_token(str.substr(start), config);
	}
}

const bpftime::agent_config& set_agent_config_from_env()
{
	bpftime::agent_config agent_config;
	if (const char *custom_helpers = getenv("BPFTIME_HELPER_GROUPS");
	    custom_helpers != nullptr) {
		agent_config.enable_kernel_helper_group =
			agent_config.enable_ffi_helper_group =
				agent_config.enable_shm_maps_helper_group =
					false;
		auto helpers_sv = std::string_view(custom_helpers);
		process_helper_sv(helpers_sv, ',', agent_config);
	} else {
		spdlog::info(
			"Enabling helper groups ffi, kernel, shm_map by default");
		agent_config.enable_kernel_helper_group =
			agent_config.enable_shm_maps_helper_group =
				agent_config.enable_ffi_helper_group = true;
	}
	const char *use_jit = getenv("BPFTIME_USE_JIT");
	agent_config.jit_enabled = use_jit != nullptr;
	bpftime_set_agent_config(agent_config);
	return bpftime_get_agent_config();
}

void start_up()
{
	if (already_setup)
		return;
	already_setup = true;
	spdlog::info("Initialize syscall server");
	spdlog::cfg::load_env_levels();
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	const auto &agent_config = set_agent_config_from_env();
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
	if (agent_config.enable_ffi_helper_group) {
		for (auto x : bpftime_helper_group::get_shm_maps_helper_group()
				      .get_helper_ids()) {
			helper_ids.push_back(x);
		}
		// non_kernel_helpers =
		for (const auto &[k, v] : get_ffi_helper_protos()) {
			non_kernel_helpers[k] = v;
		}
	}
	verifier::set_available_helpers(helper_ids);
	spdlog::info("Enabling {} helpers", helper_ids.size());
	verifier::set_non_kernel_helpers(non_kernel_helpers);
#endif
	spdlog::info("bpftime-syscall-server started");
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
		spdlog::error("Failed to open {}: {}", file, err);
		return err;
	}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	err = fscanf(f, fmt, &ret);
#pragma GCC diagnostic pop
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		spdlog::error("Failed to parse {}: {}", file, err);
		fclose(f);
		return err;
	}
	fclose(f);
	return ret;
}

int determine_uprobe_perf_type()
{
	const char *file = "/sys/bus/event_source/devices/uprobe/type";

	return parse_uint_from_file(file, "%d\n");
}

int determine_uprobe_retprobe_bit()
{
	const char *file =
		"/sys/bus/event_source/devices/uprobe/format/retprobe";

	return parse_uint_from_file(file, "config:%d\n");
}
