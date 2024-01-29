#ifndef _BPFTIME_HELPER_GROUP_HPP
#define _BPFTIME_HELPER_GROUP_HPP
#include <string>
#include <map>
#include <cinttypes>
#include <vector>
#include "bpftime_prog.hpp"

#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#endif

namespace bpftime
{
struct bpftime_helper_info {
	unsigned int index;
	std::string name;
	void *fn;
};

class bpftime_helper_group {
    public:
	bpftime_helper_group() = default;
	bpftime_helper_group(
		std::map<unsigned int, bpftime_helper_info> helper_map)
		: helper_map(helper_map)
	{
	}
	~bpftime_helper_group() = default;

	// Register a helper
	int register_helper(const bpftime_helper_info &info);

	// Append another group to the current one
	int append(const bpftime_helper_group &another_group);

	// Utility function to get the UFUNC helper group
	static const bpftime_helper_group &get_ufunc_helper_group();

	// Utility function to get the kernel utilities helper group
	static const bpftime_helper_group &get_kernel_utils_helper_group();

	// Function to register and create a local hash map helper group
	static const bpftime_helper_group &get_shm_maps_helper_group();

	// Add the helper group to the program
	int add_helper_group_to_prog(bpftime_prog *prog) const;
	// Get all helper ids of this helper group
	std::vector<int32_t> get_helper_ids() const;

    private:
	// Map to store helpers indexed by their unique ID
	std::map<unsigned int, bpftime_helper_info> helper_map;
};
#ifdef ENABLE_BPFTIME_VERIFIER
std::map<int32_t, bpftime::verifier::BpftimeHelperProrotype>
get_ufunc_helper_protos();
#endif
} // namespace bpftime
#endif
