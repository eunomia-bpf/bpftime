#ifndef _BPFTIME_FRIDA_ATTACH_PRIVATE_DATA_HPP
#define _BPFTIME_FRIDA_ATTACH_PRIVATE_DATA_HPP
#include "attach_private_data.hpp"
#include <cstdint>
#include <string>
namespace bpftime
{
namespace attach
{
// Private data for frida uprobe attach
struct frida_attach_private_data final : public attach_private_data {
    // The address to hook
	uint64_t addr;
    // Saved module name
    std::string module_name;
    // The input string should be: Either an decimal integer in string format, indicating the function address to hook. Or in format of NAME:OFFSET, where NAME is the module name (empty is ok), OFFSET is the module offset
	int initialize_from_string(const std::string_view &sv) override;
};
} // namespace attach
} // namespace bpftime

#endif
