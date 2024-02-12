#ifndef _BPFTIME_FRIDA_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_FRIDA_ATTACH_PRIVATE_DATA_HPP
#include "attach_private_data.hpp"
namespace bpftime
{
namespace attach
{
struct frida_attach_private_data final : public attach_private_data {
	uint64_t addr;
	int initialize_from_string(const std::string_view &sv);
};
} // namespace attach
} // namespace bpftime

#endif
