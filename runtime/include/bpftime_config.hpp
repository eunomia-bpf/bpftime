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
	bool enable_ffi_helper_group = false;
	bool enable_shm_maps_helper_group = true;
};
} // namespace bpftime

#endif
