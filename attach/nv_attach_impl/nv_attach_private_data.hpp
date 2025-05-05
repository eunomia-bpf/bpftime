#ifndef _NV_ATTACH_PRIVATE_DATA_HPP
#define _NV_ATTACH_PRIVATE_DATA_HPP

#include "attach_private_data.hpp"
#include <variant>
namespace bpftime
{
namespace attach
{
struct nv_attach_private_data final : public attach_private_data {

	std::variant<uintptr_t, std::string> code_addr_or_func_name;
	uintptr_t comm_shared_mem = 0;
	std::string trampoline_ptx;
	int initialize_from_string(const std::string_view &sv) override;
	std::string to_string() const override;
};

} // namespace attach
} // namespace bpftime

#endif
