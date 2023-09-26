#ifndef _VERIFIER_INTERFACE_HPP
#define _VERIFIER_INTERFACE_HPP
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
namespace bpftime
{
    // Returns none if no error occurs
std::optional<std::string> do_verification(const uint8_t *insn,
					   size_t insn_cnt);
}
#endif
