#include "frida_attach_utils.hpp"
#include <frida_uprobe_attach_impl.hpp>
#include "catch2/catch_test_macros.hpp"
#include <cstdint>
#include <dlfcn.h>

// PC/LR canonicalization exists only on ARM64: on x86_64 the Frida context
// already reports the probed address in RIP.
#if defined(__aarch64__) || defined(_M_ARM64)

using namespace bpftime;
using namespace bpftime::attach;

extern "C" __attribute__((noinline)) uint64_t
__test_arm64_canonical_pc_target(uint64_t x)
{
	// Keep a real frame and forbid inlining so the symbol has a stable,
	// instrumentable entry.
	asm("");
	return x + 1;
}
// Kernel uprobes report the probed instruction address in PT_REGS_IP and a
// canonical link register. This test pins that contract on the Frida probe
// path: the callback must observe the probed function's own entry address in
// regs.pc (never Frida's inline-hook trampoline) and, on Apple arm64e,
// PAC-free PC and LR (x30). It guards against changes in Frida's
// GumInvocationContext PC population across versions or attach backends.
TEST_CASE("ARM64 uprobe callback reports canonical function PC and LR")
{
	frida_attach_impl man;
	void *func_addr =
		find_function_addr_by_name("__test_arm64_canonical_pc_target");
	REQUIRE(func_addr != nullptr);

	uint64_t observed_pc = 0;
	uint64_t observed_lr = 0;
	int fired = 0;
	int id = man.create_uprobe_at(func_addr, [&](const pt_regs &regs) {
		observed_pc = regs.pc;
		observed_lr = regs.regs[30];
		fired++;
	});
	REQUIRE(id >= 0);

	uint64_t ret = __test_arm64_canonical_pc_target(41);
	REQUIRE(ret == 42);
	REQUIRE(fired == 1);

	// The reported PC must be the probed function's canonical entry, i.e.
	// the same address the probe was attached to, not Frida's trampoline.
	uintptr_t expected_pc = reinterpret_cast<uintptr_t>(func_addr);
#if defined(__APPLE__)
	// The attach address may be an arm64e signed pointer; compare against
	// its canonical form. xpaci is a no-op on an unsigned address.
	asm volatile("xpaci %0" : "+r"(expected_pc));
#endif
	REQUIRE(observed_pc == static_cast<uint64_t>(expected_pc));

	// Independent corroboration: the PC resolves inside a file-backed image
	// mapping. Frida's trampoline slab is an anonymous mapping for which
	// dladdr returns no match.
	Dl_info pc_info{};
	REQUIRE(dladdr(reinterpret_cast<void *>(observed_pc), &pc_info) != 0);
	REQUIRE(pc_info.dli_fname != nullptr);

#if defined(__APPLE__)
	// arm64e Pointer Authentication occupies the bits above the 39-bit
	// userspace virtual address; the reported PC and LR must both be
	// canonical with every high bit clear.
	constexpr uint64_t CANONICAL_USER_ADDR_MASK = 0x0000007fffffffffULL;
	REQUIRE((observed_pc & ~CANONICAL_USER_ADDR_MASK) == 0);
	REQUIRE((observed_lr & ~CANONICAL_USER_ADDR_MASK) == 0);
#endif
}

#endif
