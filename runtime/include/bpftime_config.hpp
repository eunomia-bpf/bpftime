#ifndef _CONFIG_MANAGER_HPP
#define _CONFIG_MANAGER_HPP

#include <cstdlib>
#include <string>

#ifndef DEFAULT_LOGGER_OUTPUT_PATH
#define DEFAULT_LOGGER_OUTPUT_PATH "~/.bpftime/runtime.log"
#endif
#define stringize(x) #x

namespace bpftime
{
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
	int shm_memory_size = 20; // 20MB

	// specify the where the logger output should be written to
	// It can be a file path or "console".
	// If it is "console", the logger will output to stderr
	std::string logger_output_path = DEFAULT_LOGGER_OUTPUT_PATH;
};

// Get the bpftime configuration from the environment variables
// If the shared memory is not int, this should be called first
const agent_config get_agent_config_from_env() noexcept;

} // namespace bpftime

#endif
