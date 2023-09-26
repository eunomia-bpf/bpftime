
#ifdef USE_EBPF_VERIFIER
#include "asm_unmarshal.hpp"
#include <crab_verifier.hpp>
#endif
#include "verifier-interface.hpp"
#include <sstream>
namespace bpftime
{
std::optional<std::string> do_verification(const uint8_t *insn, size_t insn_cnt)
{
	std::ostringstream msgbuf;
	// unmarshal()
	// std::vector<ebpf_inst>
	// unmarshal()
	return {};
}
} // namespace bpftime
