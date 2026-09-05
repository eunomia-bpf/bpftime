#include "catch2/catch_message.hpp"
#include "frida_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <frida_uprobe_attach_impl.hpp>
#include <cstdlib>
#if defined(__linux__)
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>
#endif
using namespace bpftime::attach;

extern "C" int __func_reolve_test(int a, int b)
{
	return a + b;
}

TEST_CASE("Test internal function resolve")
{
	frida_attach_impl man;
	void *addr = find_function_addr_by_name("__func_reolve_test");
	REQUIRE(addr != nullptr);
	REQUIRE(addr == (void *)&__func_reolve_test);
}

TEST_CASE("Test external function resolve")
{
	frida_attach_impl man;
	void *addr = find_function_addr_by_name("malloc");
	REQUIRE(addr != nullptr);
	INFO("malloc addr resolved: " << (uintptr_t)addr);
	REQUIRE(addr == (void *)&malloc);
}

#if defined(__linux__)
TEST_CASE("Mapped module paths survive VMA splitting")
{
	char path[] = "/tmp/bpftime-map-files-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0);
	const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
	REQUIRE(page_size > 0);
	REQUIRE(ftruncate(fd, page_size * 2) == 0);
	void *mapping = mmap(nullptr, page_size * 2, PROT_READ, MAP_PRIVATE, fd, 0);
	REQUIRE(mapping != MAP_FAILED);

	// Split the original file mapping into two VMAs. Callers may still carry a
	// map_files range describing the pre-split mapping.
	REQUIRE(mprotect(static_cast<char *>(mapping) + page_size, page_size,
			 PROT_NONE) == 0);

	char module_name[128];
	snprintf(module_name, sizeof(module_name), "/proc/%d/map_files/%lx-%lx",
		 getpid(), reinterpret_cast<uintptr_t>(mapping),
		 reinterpret_cast<uintptr_t>(mapping) + page_size * 2);
	const auto resolved = resolve_mapped_module_path(module_name);

	REQUIRE(munmap(mapping, page_size * 2) == 0);
	REQUIRE(close(fd) == 0);
	REQUIRE(unlink(path) == 0);
	REQUIRE(resolved.has_value());
	REQUIRE(*resolved == path);
}
#endif
