#include "bpftime_config.hpp"
#include "spdlog/spdlog.h"
#include <string_view>
#include <filesystem>

using namespace bpftime;

static void process_token(const std::string_view &token, agent_config &config)
{
	if (token == "ufunc") {
		SPDLOG_INFO("Enabling ufunc helper group");
		config.enable_ufunc_helper_group = true;
	} else if (token == "kernel") {
		SPDLOG_INFO("Enabling kernel helper group");
		config.enable_kernel_helper_group = true;
	} else if (token == "shm_map") {
		SPDLOG_INFO("Enabling shm_map helper group");
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

const agent_config bpftime::get_agent_config_from_env() noexcept
{
	bpftime::agent_config agent_config;
	if (const char *custom_helpers = getenv("BPFTIME_HELPER_GROUPS");
	    custom_helpers != nullptr) {
		agent_config.enable_kernel_helper_group =
			agent_config.enable_ufunc_helper_group =
				agent_config.enable_shm_maps_helper_group =
					false;
		auto helpers_sv = std::string_view(custom_helpers);
		process_helper_sv(helpers_sv, ',', agent_config);
	} else {
		agent_config.enable_kernel_helper_group =
			agent_config.enable_shm_maps_helper_group =
				agent_config.enable_ufunc_helper_group = true;
	}

	if (getenv("BPFTIME_DISABLE_JIT") != nullptr) {
		agent_config.jit_enabled = false;
	}

	if (getenv("BPFTIME_ALLOW_EXTERNAL_MAPS") != nullptr) {
		agent_config.allow_non_buildin_map_types = true;
	}

	const char *shm_memory_size_str = getenv("BPFTIME_SHM_MEMORY_MB");
	if (shm_memory_size_str != nullptr) {
		try {
			agent_config.shm_memory_size =
				std::stoi(shm_memory_size_str);
		} catch (...) {
			SPDLOG_ERROR(
				"Invalid value for BPFTIME_SHM_MEMORY_SIZE: {}",
				shm_memory_size_str);
		}
	}

	const char *logger_target = std::getenv("BPFTIME_LOG_OUTPUT");
	if (logger_target != NULL) {
		agent_config.logger_output_path = logger_target;
	}
	return agent_config;
}
