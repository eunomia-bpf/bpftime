// Miscellaneous trap backend tests: private data parsing, function address
// resolution, and SIGTRAP signal chaining to the host handler.
#include "trap_test_common.hpp"
#include "trap_attach_private_data.hpp"
#include "trap_attach_utils.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

using namespace bpftime;
using namespace bpftime::attach::trap;

// ---- Private data parsing tests ----

extern "C" TRAP_TEST_TARGET int __trap_private_data_target(int x)
{
	asm("");
	return x + 1;
}

static std::string self_exe()
{
	char buf[PATH_MAX] = {};
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	REQUIRE(n > 0);
	return std::string(buf, (size_t)n);
}

TEST_CASE("Trap backend: private data from a plain address")
{
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string("123456") == 0);
	REQUIRE(priv.addr == 123456);
	REQUIRE(priv.module_name.empty());
	REQUIRE(priv.to_string() == "123456");
}

TEST_CASE("Trap backend: private data from module and offset")
{
	auto exe = self_exe();
	void *base = get_module_base_addr("");
	REQUIRE(base != nullptr);
	REQUIRE(get_module_base_addr(exe.c_str()) == base);
	uintptr_t offset = (uintptr_t)&__trap_private_data_target -
			   (uintptr_t)base;
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string(exe + ":" +
					    std::to_string(offset)) == 0);
	REQUIRE(priv.addr == (uintptr_t)&__trap_private_data_target);
	REQUIRE(priv.module_name == exe);
}

TEST_CASE("Trap backend: private data rejects malformed input")
{
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string("/bin/ls:") == -EINVAL);
	REQUIRE(priv.initialize_from_string("not-a-number") == -EINVAL);
	REQUIRE(priv.initialize_from_string("/proc/1/map_files/0-0:1") ==
		-ENOENT);
}

TEST_CASE("Trap backend: map_files module names resolve to the mapped path")
{
	REQUIRE(resolve_mapped_module_path("/usr/lib/libc.so.6") ==
		std::string("/usr/lib/libc.so.6"));
	// A range belonging to another process cannot be resolved
	REQUIRE(!resolve_mapped_module_path("/proc/1/map_files/1000-2000"));
}

// ---- Function address resolution tests ----

// Not exported from the executable (no -rdynamic), so dlsym cannot find it
// and the ELF symbol table must be consulted.
extern "C" TRAP_TEST_TARGET int __trap_resolve_me(int x)
{
	asm("");
	return x * 3;
}

TEST_CASE("Trap backend: resolve functions of the executable by name")
{
	REQUIRE(dlsym(RTLD_DEFAULT, "__trap_resolve_me") == nullptr);
	REQUIRE(find_function_addr_by_name("__trap_resolve_me") ==
		(void *)&__trap_resolve_me);
	REQUIRE(find_function_addr_by_name("__trap_does_not_exist") == nullptr);
	REQUIRE(find_function_addr_by_name(nullptr) == nullptr);
}

TEST_CASE("Trap backend: resolve exported functions of shared libraries")
{
	void *m = find_function_addr_by_name("malloc");
	REQUIRE(m != nullptr);
	REQUIRE(m == dlsym(RTLD_DEFAULT, "malloc"));
	Dl_info info{};
	REQUIRE(dladdr(m, &info) != 0);
	REQUIRE(find_module_export_by_name(info.dli_fname, "malloc") == m);
	REQUIRE(find_module_export_by_name(info.dli_fname, "__no_such__") ==
		nullptr);
	REQUIRE(find_module_export_by_name("/no/such/lib.so", "malloc") ==
		nullptr);
	void *libc_base = get_module_base_addr(info.dli_fname);
	REQUIRE(libc_base != nullptr);
	REQUIRE(libc_base == info.dli_fbase);
	// The runtime address of an exported function equals base + offset
	// only when file offset 0 maps to the base, which holds for libc
	REQUIRE((uintptr_t)m > (uintptr_t)libc_base);
}

// ---- Signal chaining tests ----

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
