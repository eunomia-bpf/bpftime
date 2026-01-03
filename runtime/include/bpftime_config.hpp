#ifndef _CONFIG_MANAGER_HPP
#define _CONFIG_MANAGER_HPP

#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <variant>

#ifndef DEFAULT_LOGGER_OUTPUT_PATH
#define DEFAULT_LOGGER_OUTPUT_PATH "~/.bpftime/runtime.log"
#endif

#ifndef DEFAULT_VM_NAME
#define DEFAULT_VM_NAME "llvm"
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
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;

using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

// Configuration for the bpftime runtime
// Initialize the configuration from the environment variables
struct agent_config {
	bool debug = false;
	// Enable JIT?
	bool jit_enabled = true;

	// helper groups
	bool enable_kernel_helper_group = true;
	bool enable_ufunc_helper_group = false;
	bool enable_shm_maps_helper_group = true;

	// allow non builtin map types
	// if enabled, when a eBPF application tries to create a map with a
	// self-defined or non-buildin supported map type, if will not be
	// rejected
	bool allow_non_buildin_map_types = false;

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

	// Here it is a variant, since this object (agent_config) will be used
	// for both local and shared memory
	char logger_output_path[LOG_PATH_MAX_LEN] = DEFAULT_LOGGER_OUTPUT_PATH;
	char vm_name[VM_NAME_MAX_LEN] = DEFAULT_VM_NAME;

	// Optional: enable GDRCopy for host-side lookup of GPU maps.
	bool enable_gpu_gdrcopy = false;
	// If non-zero, GDRCopy is only attempted when per-key copy size <= this threshold.
	// Set to 0 to disable the threshold (always attempt when enabled).
	uint64_t gpu_gdrcopy_max_per_key_bytes =
		DEFAULT_GPU_GDRCOPY_MAX_PER_KEY_BYTES;

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

// Get the bpftime configuration from the environment variables
// If the shared memory is not int, this should be called first
agent_config construct_agent_config_from_env() noexcept;

} // namespace bpftime

#endif
