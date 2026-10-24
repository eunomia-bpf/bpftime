#ifndef _ROCM_ATTACH_PRIVATE_DATA_HPP
#define _ROCM_ATTACH_PRIVATE_DATA_HPP

#include "attach_private_data.hpp"
#include "ebpf_inst.h"
#include "base_attach_impl.hpp"
namespace bpftime
{
namespace attach
{
struct rocm_attach_private_data final : public attach_private_data {
	std::string func_name;
	uintptr_t comm_shared_mem = 0;
	std::vector<MapBasicInfo> map_basic_info;
	std::vector<ebpf_inst> instructions;
	bool is_ret_probe = false;
	int initialize_from_string(const std::string_view &sv) override;
};
} // namespace attach
} // namespace bpftime

#endif
