#include "bpftime_config.hpp"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <string_view>
#include <optional>
#include <limits>
#include <type_traits>
#include <algorithm>

using namespace bpftime;

alignas(std::atomic_ref<uint32_t>::required_alignment)
uint32_t bpftime::probe_read_window_config_snapshot;
static_assert(std::atomic_ref<uint32_t>::is_always_lock_free);
void bpftime::bpftime_set_probe_read_window_config(const runtime_config &config) noexcept
{
	const auto bytes = std::min(config.probe_read_window_bytes, 512u);
	const auto pre = std::min(config.probe_read_lookbehind_bytes, bytes);
	std::atomic_ref(probe_read_window_config_snapshot)
		.store((bytes + 1) | (pre << 13), std::memory_order_relaxed);
}

template <size_t N>
static bool copy_env(char (&out)[N], const char *name,
		     const char *fallback = "") noexcept
{
	const char *value = std::getenv(name);
	if (!value)
		value = fallback;
	const size_t len = strlen(value);
	if (len >= N) {
		out[0] = '\0';
		return false;
	}
	memcpy(out, value, len + 1);
	return true;
}

namespace bpftime
{
static agent_bootstrap_config bootstrap_config{};
static bool bootstrap_initialized = false;

void reload_agent_bootstrap_config_from_env() noexcept
{
	const int saved_errno = errno;
	bootstrap_config = {};
	copy_env(bootstrap_config.global_shm_name, "BPFTIME_GLOBAL_SHM_NAME",
		 "bpftime_maps_shm");
	bootstrap_config.invalid_process_filter = !copy_env(
		bootstrap_config.allow_process, "BPFTIME_AGENT_ALLOW_PROCESS");
	bootstrap_config.invalid_process_filter |= !copy_env(
		bootstrap_config.deny_process, "BPFTIME_AGENT_DENY_PROCESS");
	copy_env(bootstrap_config.log_output, "BPFTIME_LOG_OUTPUT");
	bootstrap_config.has_log_output =
		getenv("BPFTIME_LOG_OUTPUT") != nullptr;
	copy_env(bootstrap_config.host_procfs, "BPFTIME_HOST_PROCFS");
	if (const char *tpbase = getenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	    tpbase != nullptr && *tpbase != '\0') {
		char *end = nullptr;
		errno = 0;
		unsigned long long value = strtoull(tpbase, &end, 0);
		if (errno == 0 && end != tpbase && *end == '\0') {
			bootstrap_config.system_analysis_tpbase_offset = value;
			bootstrap_config.has_system_analysis_tpbase_offset =
				true;
		}
	}
	const char *ipc = getenv("BPFTIME_ENABLE_AGENT_IPC");
	bootstrap_config.enable_agent_ipc =
		ipc && *ipc && strcmp(ipc, "0") != 0;
	bootstrap_config.go_seccomp = getenv("BPFTIME_GO_SECCOMP") != nullptr;
	bootstrap_initialized = true;
	errno = saved_errno;
}

const agent_bootstrap_config &get_agent_bootstrap_config() noexcept
{
	if (!bootstrap_initialized)
		reload_agent_bootstrap_config_from_env();
	return bootstrap_config;
}
} // namespace bpftime
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
		} else if constexpr (std::is_unsigned_v<T>) {
			value = static_cast<T>(std::stoull(env_str));
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

static void process_token(const std::string_view &token, runtime_config &config)
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
			      runtime_config &config)
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

runtime_config bpftime::construct_runtime_config_from_env() noexcept
{
	const int saved_errno = errno;
	bpftime::runtime_config runtime_config;
	if (const char *custom_helpers = getenv("BPFTIME_HELPER_GROUPS");
	    custom_helpers != nullptr) {
		runtime_config.enable_kernel_helper_group =
			runtime_config.enable_ufunc_helper_group =
				runtime_config.enable_shm_maps_helper_group =
					false;
		auto helpers_sv = std::string_view(custom_helpers);
		process_helper_sv(helpers_sv, ',', runtime_config);
	} else {
		runtime_config.enable_kernel_helper_group =
			runtime_config.enable_shm_maps_helper_group =
				runtime_config.enable_ufunc_helper_group = true;
	}

	if (getenv("BPFTIME_DISABLE_JIT") != nullptr) {
		runtime_config.jit_enabled = false;
	}

	if (getenv("BPFTIME_ALLOW_EXTERNAL_MAPS") != nullptr) {
		runtime_config.allow_non_buildin_map_types = true;
	}

	if (getenv("BPFTIME_FRIDA_FUZZY_BACKTRACER") != nullptr) {
		runtime_config.enable_frida_fuzzy_backtracer = true;
	}

	if (auto bytes = parse_numeric_env<int>("BPFTIME_PROBE_READ_WINDOW_BYTES", 0, 512))
		runtime_config.probe_read_window_bytes = *bytes;
	if (auto pre = parse_numeric_env<int>("BPFTIME_PROBE_READ_LOOKBEHIND_BYTES", 0, 512))
		runtime_config.probe_read_lookbehind_bytes = *pre;
	runtime_config.probe_read_lookbehind_bytes =
		std::min(runtime_config.probe_read_lookbehind_bytes,
			 runtime_config.probe_read_window_bytes);

	// Parse shared memory size with validation (1MB min, 10GB max)
	if (auto shm_size = parse_numeric_env<int>("BPFTIME_SHM_MEMORY_MB", 1, 10240)) {
		runtime_config.shm_memory_size = *shm_size;
	}

	// Parse max FD count with validation
	if (auto max_fd = parse_numeric_env<size_t>("BPFTIME_MAX_FD_COUNT", 
	                                             MIN_MAX_FD_COUNT, 
	                                             MAX_MAX_FD_COUNT)) {
		runtime_config.max_fd_count = *max_fd;
	}

	const char *vm_name = std::getenv("BPFTIME_VM_NAME");

	if (vm_name != nullptr) {
		SPDLOG_INFO("Using VM: {}", vm_name);
		copy_env(runtime_config.vm_name, "BPFTIME_VM_NAME");
	}

	const char *verifier_level = std::getenv("BPFTIME_VERIFIER_LEVEL");
	if (verifier_level != nullptr) {
		std::string_view level_sv(verifier_level);
		if (level_sv == "STRICT") {
			runtime_config.verifier_mode = BPFTIME_VERIFIER_STRICT;
			SPDLOG_INFO("Verifier mode: STRICT");
		} else if (level_sv == "WARNING") {
			runtime_config.verifier_mode = BPFTIME_VERIFIER_WARNING;
			SPDLOG_INFO("Verifier mode: WARNING");
		} else if (level_sv == "NO_VERIFY") {
			runtime_config.verifier_mode = BPFTIME_NO_VERIFY;
			SPDLOG_INFO("Verifier mode: NO_VERIFY");
		} else {
			SPDLOG_WARN(
				"Unknown BPFTIME_VERIFIER_LEVEL value: {}. "
				"Valid values: STRICT, WARNING, NO_VERIFY. "
				"Defaulting to WARNING.",
				verifier_level);
		}
	}

	if (std::getenv("BPFTIME_LOG_OUTPUT") != nullptr) {
		copy_env(runtime_config.logger_output_path, "BPFTIME_LOG_OUTPUT");
	} else if (get_agent_bootstrap_config().go_seccomp) {
		runtime_config.set_logger_output_path("console");
	}

	// Optional GDRCopy knobs for GPU map host-side lookup.
	// Keep env parsing here (single place) to avoid sprinkling getenv() across modules.
	if (const char *enable_gdrcopy = std::getenv("BPFTIME_GPU_ARRAY_GDRCOPY");
	    enable_gdrcopy != nullptr && enable_gdrcopy[0] == '1') {
		runtime_config.enable_gpu_gdrcopy = true;
	}
	if (auto max_bytes = parse_numeric_env<unsigned long>(
		    "BPFTIME_GPU_ARRAY_GDRCOPY_MAX_PER_KEY_BYTES", 0ul,
		    1024ul * 1024ul * 1024ul)) {
		runtime_config.gpu_gdrcopy_max_per_key_bytes =
			static_cast<uint64_t>(*max_bytes);
	}
	if (auto v = parse_numeric_env<uint64_t>("BPFTIME_MAP_GPU_THREAD_COUNT")) {
		runtime_config.map_gpu_thread_count = *v;
		runtime_config.has_map_gpu_thread_count = true;
	}
	errno = saved_errno;
	return runtime_config;
}
