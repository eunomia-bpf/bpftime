#include "bpftime_config.hpp"
#include <cerrno>
#include <algorithm>
#include <limits>
#include <optional>
#include <string_view>
#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern "C" char **environ;
#endif

namespace bpftime
{
namespace
{
constinit agent_bootstrap_config bootstrap_config{};
constinit bool bootstrap_initialized = false;
constinit bool local_initialized = false;

char **startup_environment() noexcept
{
#if defined(__APPLE__)
	return *_NSGetEnviron();
#else
	return environ;
#endif
}

// Never retain pointers into environ. Oversized values become empty, rather
// than a truncated path or partial process-name token.
template <size_t N> bool copy_value(char (&out)[N], const char *value) noexcept
{
	const size_t len = strlen(value);
	out[0] = '\0';
	if (len >= N)
		return false;
	memcpy(out, value, len + 1);
	return true;
}

template <typename T>
std::optional<T> number(const char *text, int base = 10,
			T min = std::numeric_limits<T>::min(),
			T max = std::numeric_limits<T>::max()) noexcept
{
	char *end = nullptr;
	errno = 0;
	if constexpr (std::is_unsigned_v<T>) {
		const auto value = strtoull(text, &end, base);
		if (errno || end == text || (base == 0 && *end))
			return {};
		return static_cast<T>(std::clamp(value, (unsigned long long)min,
						 (unsigned long long)max));
	} else {
		const auto value = strtoll(text, &end, base);
		if (errno || end == text || (base == 0 && *end) ||
		    value < std::numeric_limits<T>::min() ||
		    value > std::numeric_limits<T>::max())
			return {};
		return static_cast<T>(
			std::clamp(value, (long long)min, (long long)max));
	}
}
} // namespace

// Explicit zero values keep this live, trivially destructible object in .bss.
// Runtime defaults are installed at the accepted startup entry.
constinit runtime_config local_runtime_config_snapshot{
	.jit_enabled = false,
	.enable_kernel_helper_group = false,
	.enable_shm_maps_helper_group = false,
	.shm_memory_size = 0,
	.max_fd_count = 0,
	.logger_output_path = {},
	.vm_name = {},
	.gpu_gdrcopy_max_per_key_bytes = 0,
	.system_analysis_ptregs_offset = 0,
};

void reload_agent_bootstrap_config_from_env() noexcept
{
	const int saved_errno = errno;
	bootstrap_config = {};
	copy_value(bootstrap_config.global_shm_name, "bpftime_maps_shm");
	for (char **p = startup_environment(); p && *p; ++p) {
		const char *equal = strchr(*p, '=');
		if (!equal)
			continue;
		const std::string_view name(*p, equal - *p);
		const char *value = equal + 1;
		if (name == "BPFTIME_GLOBAL_SHM_NAME")
			copy_value(bootstrap_config.global_shm_name, value);
		else if (name == "BPFTIME_AGENT_ALLOW_PROCESS")
			bootstrap_config.invalid_process_filter |= !copy_value(
				bootstrap_config.allow_process, value);
		else if (name == "BPFTIME_AGENT_DENY_PROCESS")
			bootstrap_config.invalid_process_filter |= !copy_value(
				bootstrap_config.deny_process, value);
		else if (name == "BPFTIME_ENABLE_AGENT_IPC")
			bootstrap_config.enable_agent_ipc =
				*value && strcmp(value, "0") != 0;
		else if (name == "BPFTIME_LOG_OUTPUT") {
			bootstrap_config.has_log_output = true;
			copy_value(bootstrap_config.log_output, value);
		} else if (name == "BPFTIME_GO_SECCOMP")
			bootstrap_config.go_seccomp = true;
	}
	bootstrap_initialized = true;
	errno = saved_errno;
}

const agent_bootstrap_config &get_agent_bootstrap_config() noexcept
{
	if (!bootstrap_initialized)
		reload_agent_bootstrap_config_from_env();
	return bootstrap_config;
}

void bpftime_set_local_runtime_config(const runtime_config &config) noexcept
{
	local_runtime_config_snapshot = config;
	local_initialized = true;
}

void initialize_local_runtime_config_from_env() noexcept
{
	if (!local_initialized)
		bpftime_set_local_runtime_config(
			construct_runtime_config_from_env());
}

runtime_config construct_runtime_config_from_env() noexcept
{
	const int saved_errno = errno;
	runtime_config config;
	config.enable_kernel_helper_group = config.enable_ufunc_helper_group =
		config.enable_shm_maps_helper_group = true;
	for (char **p = startup_environment(); p && *p; ++p) {
		const char *equal = strchr(*p, '=');
		if (!equal)
			continue;
		const std::string_view name(*p, equal - *p);
		const char *value = equal + 1;
		if (name == "BPFTIME_HELPER_GROUPS") {
			config.enable_kernel_helper_group =
				config.enable_ufunc_helper_group =
					config.enable_shm_maps_helper_group =
						false;
			std::string_view list(value);
			while (!list.empty()) {
				auto end = list.find(',');
				std::string_view token(
					list.data(),
					end == list.npos ? list.size() : end);
				if (token == "kernel")
					config.enable_kernel_helper_group =
						true;
				if (token == "ufunc")
					config.enable_ufunc_helper_group = true;
				if (token == "shm_map")
					config.enable_shm_maps_helper_group =
						true;
				if (end == list.npos)
					break;
				list.remove_prefix(end + 1);
			}
		} else if (name == "BPFTIME_DISABLE_JIT")
			config.jit_enabled = false;
		else if (name == "BPFTIME_ALLOW_EXTERNAL_MAPS")
			config.allow_non_buildin_map_types = true;
		else if (name == "BPFTIME_FRIDA_FUZZY_BACKTRACER")
			config.enable_frida_fuzzy_backtracer = true;
		else if (name == "BPFTIME_SHM_MEMORY_MB") {
			if (auto v = number<int>(value, 10, 1, 10240))
				config.shm_memory_size = *v;
		} else if (name == "BPFTIME_MAX_FD_COUNT") {
			if (auto v = number<size_t>(value, 10, MIN_MAX_FD_COUNT,
						    MAX_MAX_FD_COUNT))
				config.max_fd_count = *v;
		} else if (name == "BPFTIME_VERIFIER_LEVEL") {
			if (strcmp(value, "STRICT") == 0)
				config.verifier_mode = BPFTIME_VERIFIER_STRICT;
			else if (strcmp(value, "NO_VERIFY") == 0)
				config.verifier_mode = BPFTIME_NO_VERIFY;
		} else if (name == "BPFTIME_GPU_ARRAY_GDRCOPY")
			config.enable_gpu_gdrcopy = value[0] == '1';
		else if (name ==
			 "BPFTIME_GPU_ARRAY_GDRCOPY_MAX_PER_KEY_BYTES") {
			if (auto v = number<uint64_t>(value, 10, 0,
						      1024ULL * 1024 * 1024))
				config.gpu_gdrcopy_max_per_key_bytes = *v;
		} else if (name == "BPFTIME_RUN_WITH_KERNEL")
			config.run_with_kernel = true;
		else if (name == "BPFTIME_CUDA_DISABLE_CUOBJDUMP")
			config.cuda_disable_cuobjdump = value[0] == '1';
		else if (name == "BPFTIME_CUDA_DEVICE") {
			if (auto v = number<int>(value, 10, 0)) {
				config.cuda_device = *v;
				config.has_cuda_device = true;
			}
		} else if (name == "BPFTIME_MAP_GPU_THREAD_COUNT") {
			if (auto v = number<uint64_t>(value)) {
				config.map_gpu_thread_count = *v;
				config.has_map_gpu_thread_count = true;
			}
		} else if (name == "BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS") {
			if (auto v = number<uint64_t>(value, 0)) {
				config.system_analysis_tpbase_address = *v;
				config.has_system_analysis_tpbase_address =
					true;
			}
		} else if (name == "BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET") {
			if (auto v = number<uint64_t>(value, 0)) {
				config.system_analysis_tpbase_offset = *v;
				config.has_system_analysis_tpbase_offset = true;
			}
		} else if (name == "BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET") {
			if (auto v = number<uint64_t>(value, 0))
				config.system_analysis_ptregs_offset = *v;
		}
#define BPFTIME_STRING_ENV(env, field)                                         \
	else if (name == env) copy_value(config.field, value)
		BPFTIME_STRING_ENV("BPFTIME_VM_NAME", vm_name);
		BPFTIME_STRING_ENV("BPFTIME_LOG_OUTPUT", logger_output_path);
		BPFTIME_STRING_ENV("BPFTIME_HOST_PROCFS", host_procfs);
		BPFTIME_STRING_ENV("BPFTIME_SM_ARCH", sm_arch);
		BPFTIME_STRING_ENV("BPFTIME_NOT_LOAD_PATTERN",
				   not_load_pattern);
		BPFTIME_STRING_ENV("BPFTIME_KALLSYMS_PATH", kallsyms_path);
		BPFTIME_STRING_ENV("BPFTIME_BTF_PATH", btf_path);
		BPFTIME_STRING_ENV("BPFTIME_PTXPASS_LIBRARIES",
				   ptxpass_libraries);
		BPFTIME_STRING_ENV("BPFTIME_PTX_COMPILER_LIB",
				   ptx_compiler_lib);
		BPFTIME_STRING_ENV("BPFTIME_CUDA_LATE_PTX_DIR",
				   cuda_late_ptx_dir);
		BPFTIME_STRING_ENV("BPFTIME_CUOBJDUMP", cuobjdump);
		BPFTIME_STRING_ENV("BPFTIME_CUDA_ROOT", cuda_roots[0]);
		BPFTIME_STRING_ENV("CUDA_HOME", cuda_roots[1]);
		BPFTIME_STRING_ENV("CUDA_PATH", cuda_roots[2]);
		BPFTIME_STRING_ENV("LLVMBPF_CUDA_PATH", cuda_roots[3]);
		BPFTIME_STRING_ENV("CUDAToolkit_ROOT", cuda_roots[4]);
		BPFTIME_STRING_ENV("HOME", home);
		BPFTIME_STRING_ENV("TMPDIR", tmpdir);
		BPFTIME_STRING_ENV("SPDLOG_LEVEL", log_levels);
#undef BPFTIME_STRING_ENV
	}
	const auto &bootstrap = get_agent_bootstrap_config();
	if (bootstrap.go_seccomp && !bootstrap.has_log_output)
		config.set_logger_output_path("console");
	errno = saved_errno;
	return config;
}
} // namespace bpftime
