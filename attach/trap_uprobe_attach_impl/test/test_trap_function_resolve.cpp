#include "trap_test_common.hpp"
#include "trap_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <dlfcn.h>

using namespace bpftime::attach::trap;

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
