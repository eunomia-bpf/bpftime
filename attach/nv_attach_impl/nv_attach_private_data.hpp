#ifndef _NV_ATTACH_PRIVATE_DATA_HPP
#define _NV_ATTACH_PRIVATE_DATA_HPP

#include "attach_private_data.hpp"
#include "nv_attach_impl.hpp"
#include <variant>
#include <ebpf_inst.h>
#include <vector>
namespace bpftime
{
namespace attach
{
struct nv_attach_private_data final : public attach_private_data {
	std::variant<uintptr_t, std::string> code_addr_or_func_name;
	// Names of kernels to be patched when multi-stream is used
	std::vector<std::string> func_names;
	uintptr_t comm_shared_mem = 0;
	std::vector<MapBasicInfo> map_basic_info;
	std::vector<ebpf_inst> instructions;
	int initialize_from_string(const std::string_view &sv) override;
	std::string to_string() const override;
};

} // namespace attach
} // namespace bpftime

#endif
