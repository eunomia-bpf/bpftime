#ifndef _CONFIG_MANAGER_HPP
#define _CONFIG_MANAGER_HPP

namespace bpftime
{
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
	int shm_memory_size = 50; // 50MB
};

inline const agent_config get_agent_config_from_env()
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
		SPDLOG_INFO(
			"Enabling helper groups ufunc, kernel, shm_map by default");
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
	const char* shm_memory_size_str = getenv("BPFTIME_SHM_MEMORY_MB");
	if (shm_memory_size_str != nullptr) {
		try {
			agent_config.shm_memory_size = std::stoi(shm_memory_size_str);
		} catch (...) {
			SPDLOG_ERROR(
				"Invalid value for BPFTIME_SHM_MEMORY_SIZE: {}",
				shm_memory_size_str);
		}
	}
	return agent_config;
}

} // namespace bpftime

#endif
