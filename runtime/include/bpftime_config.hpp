#ifndef _CONFIG_MANAGER_HPP
#define _CONFIG_MANAGER_HPP

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <type_traits>

#ifndef DEFAULT_LOGGER_OUTPUT_PATH
#define DEFAULT_LOGGER_OUTPUT_PATH "~/.bpftime/runtime.log"
#endif

#ifndef DEFAULT_VM_NAME
#if defined(BPFTIME_LLVM_JIT)
#define DEFAULT_VM_NAME "llvm"
#elif defined(BPFTIME_UBPF_JIT)
#define DEFAULT_VM_NAME "ubpf"
#else
#define DEFAULT_VM_NAME ""
#endif
#endif
#define LOG_PATH_MAX_LEN 1024
#define VM_NAME_MAX_LEN 128

// Default and limits for file descriptor count
constexpr size_t DEFAULT_MAX_FD_COUNT = 1024 * 6;
constexpr size_t MIN_MAX_FD_COUNT = 128;
constexpr size_t MAX_MAX_FD_COUNT = 1024 * 1024; // 1M max
constexpr uint64_t DEFAULT_GPU_GDRCOPY_MAX_PER_KEY_BYTES = 4096;

namespace bpftime
{
// Verification mode for bpftime userspace verifier
// Controlled via BPFTIME_VERIFIER_LEVEL environment variable
enum bpftime_verifier_mode {
	// Userspace verifier failure is treated as a warning (default)
	// Program loading continues even if verification fails
	BPFTIME_VERIFIER_WARNING = 0,
	// Userspace verifier failure is treated as an error
	// Program loading is rejected if verification fails
	BPFTIME_VERIFIER_STRICT = 1,
	// No verification is performed
	BPFTIME_NO_VERIFY = 2,
};

// Process-local startup configuration captured once before SHM/logging.
struct agent_bootstrap_config {
	char global_shm_name[256];
	char allow_process[1024];
	char deny_process[1024];
	char log_output[LOG_PATH_MAX_LEN];
	char host_procfs[1024];
	uint64_t system_analysis_tpbase_offset;
	bool has_system_analysis_tpbase_offset;
	bool has_log_output;
	bool enable_agent_ipc;
	bool go_seccomp;
	bool invalid_process_filter;
};
static_assert(std::is_trivially_copyable_v<agent_bootstrap_config>);
const agent_bootstrap_config &get_agent_bootstrap_config() noexcept;
void reload_agent_bootstrap_config_from_env() noexcept;

// Configuration for the bpftime runtime
// Initialize the configuration from the environment variables
struct runtime_config {
	bool debug = false;
	// Enable JIT?
	bool jit_enabled = true;

	// helper groups
	bool enable_kernel_helper_group = true;
	bool enable_ufunc_helper_group = false;
	bool enable_shm_maps_helper_group = true;

	// Userspace verifier mode (see bpftime_verifier_mode enum)
	// Default is BPFTIME_VERIFIER_WARNING: verification failure is a warning
	bpftime_verifier_mode verifier_mode = BPFTIME_VERIFIER_WARNING;

	// allow non builtin map types
	// if enabled, when a eBPF application tries to create a map with a
	// self-defined or non-buildin supported map type, if will not be
	// rejected
	bool allow_non_buildin_map_types = false;

	// Use Frida's fuzzy backtracer for user stack collection. Accurate
	// unwinding remains the default.
	bool enable_frida_fuzzy_backtracer = false;

	// memory size will determine the maximum size of the shared memory
	// available for the eBPF programs and maps
	// The value is in MB
	int shm_memory_size = 50; // default to 50MB

	// Maximum file descriptor count for handler manager
	// This determines the size of the handler vector
	// Default is DEFAULT_MAX_FD_COUNT (6144)
	size_t max_fd_count = DEFAULT_MAX_FD_COUNT;

	// specify the where the logger output should be written to
	// It can be a file path or "console".
	// If it is "console", the logger will output to stderr

	// Here it is a variant, since this object (runtime_config) will be used
	// for both local and shared memory
	char logger_output_path[LOG_PATH_MAX_LEN] = DEFAULT_LOGGER_OUTPUT_PATH;
	char vm_name[VM_NAME_MAX_LEN] = DEFAULT_VM_NAME;

	// Optional: enable GDRCopy for host-side lookup of GPU maps.
	bool enable_gpu_gdrcopy = false;
	// If non-zero, GDRCopy is only attempted when per-key copy size <= this threshold.
	// Set to 0 to disable the threshold (always attempt when enabled).
	uint64_t gpu_gdrcopy_max_per_key_bytes =
		DEFAULT_GPU_GDRCOPY_MAX_PER_KEY_BYTES;

	uint64_t map_gpu_thread_count = 0;
	bool has_map_gpu_thread_count = false;

	const char *get_logger_output_path() const
	{
		return logger_output_path;
	}
	void set_logger_output_path(const char *path)
	{
		strcpy(logger_output_path, path);
	}
	void set_vm_name(const char *name)
	{
		strcpy(vm_name, name);
	}
	const char *get_vm_name() const
	{
		return vm_name;
	}
};

static_assert(std::is_trivially_copyable_v<runtime_config>);

// Get the bpftime configuration from the environment variables
// If the shared memory is not int, this should be called first
runtime_config construct_runtime_config_from_env() noexcept;

} // namespace bpftime

#endif
