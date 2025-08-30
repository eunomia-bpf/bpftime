#include "bpftime_config.hpp"
#include "spdlog/spdlog.h"
#include <string_view>
#include <optional>
#include <limits>
#include <type_traits>

using namespace bpftime;

// Helper function to parse and validate numeric environment variables
template<typename T>
static std::optional<T> parse_numeric_env(const char* env_name, 
                                          T min_value = std::numeric_limits<T>::min(),
                                          T max_value = std::numeric_limits<T>::max())
{
	const char* env_str = std::getenv(env_name);
	if (env_str == nullptr) {
		return std::nullopt;
	}

	try {
		// Parse based on type
		T value;
		if constexpr (std::is_same_v<T, int>) {
			value = std::stoi(env_str);
		} else if constexpr (std::is_same_v<T, long>) {
			value = std::stol(env_str);
		} else if constexpr (std::is_same_v<T, size_t> || std::is_same_v<T, unsigned long>) {
			value = std::stoul(env_str);
		} else {
			// Fallback for other types
			value = static_cast<T>(std::stoll(env_str));
		}
		
		if (value < min_value) {
			SPDLOG_WARN("{} value {} is below minimum {}, using minimum",
			           env_name, value, min_value);
			return min_value;
		}
		
		if (value > max_value) {
			SPDLOG_WARN("{} value {} exceeds maximum {}, using maximum",
			           env_name, value, max_value);
			return max_value;
		}
		
		SPDLOG_INFO("Setting {} to: {}", env_name, value);
		return value;
	} catch (...) {
		SPDLOG_ERROR("Invalid value for {}: {}", env_name, env_str);
		return std::nullopt;
	}
}

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
		SPDLOG_WARN("Unknown helper group: {}", token);
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

agent_config bpftime::construct_agent_config_from_env() noexcept
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

	// Parse shared memory size with validation (1MB min, 10GB max)
	if (auto shm_size = parse_numeric_env<int>("BPFTIME_SHM_MEMORY_MB", 1, 10240)) {
		agent_config.shm_memory_size = *shm_size;
	}

	// Parse max FD count with validation
	if (auto max_fd = parse_numeric_env<size_t>("BPFTIME_MAX_FD_COUNT", 
	                                             MIN_MAX_FD_COUNT, 
	                                             MAX_MAX_FD_COUNT)) {
		agent_config.max_fd_count = *max_fd;
	}

	const char *vm_name = std::getenv("BPFTIME_VM_NAME");

	if (vm_name != nullptr) {
		SPDLOG_INFO("Using VM: {}", vm_name);
		agent_config.set_vm_name(vm_name);
	}

	const char *logger_target = std::getenv("BPFTIME_LOG_OUTPUT");
	if (logger_target != nullptr) {
		agent_config.set_logger_output_path(logger_target);
	}
	return agent_config;
}
