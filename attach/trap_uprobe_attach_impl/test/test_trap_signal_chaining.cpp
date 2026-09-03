#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <csignal>
#include <cstdint>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t __trap_chain_target(uint64_t a)
{
	asm("");
	return a + 100;
}

static volatile sig_atomic_t host_handler_hits = 0;

static void host_sigtrap_handler(int, siginfo_t *, void *)
{
	host_handler_hits = host_handler_hits + 1;
}

// The host process may already own SIGTRAP. Traps that are not ours must be
// forwarded to it, and detaching must leave its handler in place.
TEST_CASE("Trap backend: chains to a pre-existing SIGTRAP handler")
{
	struct sigaction host {};
	host.sa_sigaction = host_sigtrap_handler;
	host.sa_flags = SA_SIGINFO;
	sigemptyset(&host.sa_mask);
	struct sigaction saved {};
	REQUIRE(sigaction(SIGTRAP, &host, &saved) == 0);

	host_handler_hits = 0;
	int probe_hits = 0;
	{
		trap_attach_impl man;
		REQUIRE(man.create_uprobe_at((void *)&__trap_chain_target,
					     [&](const pt_regs &) {
						     probe_hits++;
					     }) >= 0);
		REQUIRE(__trap_chain_target(1) == 101);
		REQUIRE(probe_hits == 1);
		REQUIRE(host_handler_hits == 0);
		// A SIGTRAP that does not come from one of our breakpoints
		// reaches the host handler
		raise(SIGTRAP);
		REQUIRE(host_handler_hits == 1);
		REQUIRE(__trap_chain_target(2) == 102);
		REQUIRE(probe_hits == 2);
	}
	// After detaching, the function runs untouched and the host handler
	// still receives its signals
	REQUIRE(__trap_chain_target(3) == 103);
	REQUIRE(probe_hits == 2);
	raise(SIGTRAP);
	REQUIRE(host_handler_hits == 2);
	REQUIRE(sigaction(SIGTRAP, &saved, nullptr) == 0);
}
