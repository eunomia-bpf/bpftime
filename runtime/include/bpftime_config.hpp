#ifndef _CONFIG_MANAGER_HPP
#define _CONFIG_MANAGER_HPP

#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib>
#include <string>
#include <variant>

#ifndef DEFAULT_LOGGER_OUTPUT_PATH
#define DEFAULT_LOGGER_OUTPUT_PATH "~/.bpftime/runtime.log"
#endif
#define stringize(x) #x

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
	int shm_memory_size = 20; // 20MB

	// specify the where the logger output should be written to
	// It can be a file path or "console".
	// If it is "console", the logger will output to stderr

	// Here it is a variant, since this object (agent_config) will be used
	// for both local and shared memory
	std::variant<std::string, boost_shm_string> logger_output_path;

	agent_config(boost::interprocess::managed_shared_memory &memory)
		: logger_output_path(
			  boost_shm_string(memory.get_segment_manager()))
	{
		std::get<boost_shm_string>(logger_output_path) =
			DEFAULT_LOGGER_OUTPUT_PATH;
	}
	agent_config()
		: logger_output_path(std::string(DEFAULT_LOGGER_OUTPUT_PATH))
	{
	}
	const char *get_logger_output_path() const
	{
		return std::visit(
			[](auto &&arg) -> const char * { return arg.c_str(); },
			logger_output_path);
	}
	void set_logger_output_path(const char *path)
	{
		std::visit([&](auto &&arg) { arg = path; }, logger_output_path);
	}
	void
	change_to_shm_object(boost::interprocess::managed_shared_memory &memory)
	{
		if (std::holds_alternative<boost_shm_string>(
			    logger_output_path))
			return;
		auto current_value = std::get<std::string>(logger_output_path);
		logger_output_path.emplace<boost_shm_string>(
			current_value.c_str(), memory.get_segment_manager());
	}
	agent_config(const agent_config &) = delete;
	agent_config &operator=(const agent_config &) = delete;
	agent_config(agent_config &&) noexcept = default;
	agent_config &operator=(agent_config &&) noexcept = default;
};

// Get the bpftime configuration from the environment variables
// If the shared memory is not int, this should be called first
agent_config construct_agent_config_from_env() noexcept;

} // namespace bpftime

#endif
