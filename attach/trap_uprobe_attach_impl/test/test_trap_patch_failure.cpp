#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

using namespace bpftime;
using namespace bpftime::attach::trap;

namespace {
uintptr_t watched_begin = 0, watched_end = 0;
int deny_write = 0, deny_restore = 0;
bool deny_rollback = false;
bool deny_rwx = false;

bool consume_failure(int &remaining)
{
	if (remaining == 0)
		return false;
	if (remaining > 0)
		--remaining;
	return true;
}

struct fault_scope {
	~fault_scope()
	{
		watched_begin = watched_end = 0;
		deny_write = deny_restore = 0;
		deny_rollback = false;
		deny_rwx = false;
	}
};

int page_prot(void *addr)
{
	FILE *file = fopen("/proc/self/maps", "r");
	if (!file)
		return -1;
	char line[1024];
	int result = -1;
	while (fgets(line, sizeof(line), file)) {
		unsigned long lo, hi;
		char perms[5];
		if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) == 3 &&
		    (uintptr_t)addr >= lo && (uintptr_t)addr < hi) {
			result = (perms[0] == 'r' ? PROT_READ : 0) |
				 (perms[1] == 'w' ? PROT_WRITE : 0) |
				 (perms[2] == 'x' ? PROT_EXEC : 0);
			break;
		}
	}
	fclose(file);
	return result;
}

struct target_code {
	size_t page = (size_t)sysconf(_SC_PAGESIZE);
	uint8_t *mapping;
	uint8_t *entry;
	uint32_t original[2] = { 0x00150513, 0x00008067 }; // addi a0,a0,1; ret

	explicit target_code(bool cross_page = false)
	{
		mapping = (uint8_t *)mmap(nullptr, 2 * page,
			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		REQUIRE(mapping != MAP_FAILED);
		entry = mapping + (cross_page ? page - 2 : 0);
		memcpy(entry, original, sizeof(original));
		__builtin___clear_cache((char *)entry, (char *)entry + sizeof(original));
		REQUIRE(mprotect(mapping, 2 * page, PROT_READ | PROT_EXEC) == 0);
		watched_begin = (uintptr_t)mapping;
		watched_end = watched_begin + 2 * page;
	}
	// The engine retains decoded sites for its lifetime. Keep test mappings
	// alive too, including when a deliberately failed rollback leaves a trap.
	uint64_t call() const { return ((uint64_t (*)(uint64_t))entry)(41); }
	bool restored() const { return memcmp(entry, original, sizeof(original)) == 0; }
};
} // namespace

extern "C" int __real_mprotect(void *, size_t, int);
extern "C" int __wrap_mprotect(void *addr, size_t len, int prot)
{
	if ((uintptr_t)addr >= watched_begin && (uintptr_t)addr < watched_end) {
		bool fail = (prot & PROT_WRITE) ? consume_failure(deny_write) :
			consume_failure(deny_restore);
		fail |= deny_rwx && (prot & PROT_WRITE) && (prot & PROT_EXEC);
		if (fail) {
			if (!(prot & PROT_WRITE) && deny_rollback)
				deny_write = -1;
			errno = EACCES;
			return -1;
		}
	}
	return __real_mprotect(addr, len, prot);
}

TEST_CASE("Trap backend: failed RX restoration rolls back without writing RX")
{
	fault_scope faults;
	target_code code;
	trap_attach_impl man;
	int hits = 0;
	deny_restore = 1; // The retry succeeds; rollback must reacquire WRITE.
	REQUIRE(man.create_uprobe_at(code.entry, [&](const pt_regs &) { ++hits; }) == -EIO);
	REQUIRE(code.restored());
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_EXEC));
	REQUIRE(code.call() == 42);
	REQUIRE(hits == 0);
}

TEST_CASE("Trap backend: W^X rejection never removes EXEC from host code")
{
	fault_scope faults;
	target_code code;
	trap_attach_impl man;
	deny_rwx = true;
	REQUIRE(man.create_uprobe_at(code.entry, [](const pt_regs &) {}) == -EIO);
	REQUIRE(code.restored());
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_EXEC));
	REQUIRE(code.call() == 42);
}

TEST_CASE("Trap backend: persistent RX failure is reported and retry repairs permissions")
{
	fault_scope faults;
	target_code code;
	trap_attach_impl man;
	deny_restore = -1;
	REQUIRE(man.create_uprobe_at(code.entry, [](const pt_regs &) {}) == -EIO);
	REQUIRE(code.restored());
	REQUIRE(code.call() == 42);
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_WRITE | PROT_EXEC));
	deny_restore = 0;
	const int id = man.create_uprobe_at(code.entry, [](const pt_regs &) {});
	REQUIRE(id >= 0);
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_EXEC));
	REQUIRE(man.detach_by_id(id) == 0);
}

TEST_CASE("Trap backend: failed rollback leaves a callback-free executable site")
{
	fault_scope faults;
	target_code code;
	trap_attach_impl man;
	int hits = 0;
	deny_restore = 1;
	deny_rollback = true;
	REQUIRE(man.create_uprobe_at(code.entry, [&](const pt_regs &) { ++hits; }) == -EIO);
	REQUIRE(code.call() == 42);
	REQUIRE(hits == 0);
	deny_write = 0;
	deny_rollback = false;
	const int id = man.create_uprobe_at(code.entry, [&](const pt_regs &) { ++hits; });
	REQUIRE(id >= 0);
	REQUIRE(code.call() == 42);
	REQUIRE(hits == 1);
	REQUIRE(man.detach_by_id(id) == 0);
	REQUIRE(code.restored());
}

TEST_CASE("Trap backend: failed disarm remains retryable")
{
	fault_scope faults;
	target_code code;
	trap_attach_impl man;
	int hits = 0;
	const int id = man.create_uprobe_at(code.entry, [&](const pt_regs &) { ++hits; });
	REQUIRE(id >= 0);
	deny_write = -1;
	REQUIRE(man.detach_by_id(id) == -EIO);
	REQUIRE(code.call() == 42);
	REQUIRE(hits == 1);
	deny_write = 0;
	deny_restore = 1;
	REQUIRE(man.detach_by_id(id) == -EIO);
	REQUIRE(code.restored());
	REQUIRE(code.call() == 42);
	REQUIRE(hits == 1);
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_EXEC));
}

TEST_CASE("Trap backend: cross-page patch preserves each page's permissions")
{
	fault_scope faults;
	target_code code(true);
	REQUIRE(mprotect(code.mapping + code.page, code.page,
		PROT_READ | PROT_WRITE | PROT_EXEC) == 0);
	trap_attach_impl man;
	const int id = man.create_uprobe_at(code.entry, [](const pt_regs &) {});
	REQUIRE(id >= 0);
	REQUIRE(code.call() == 42);
	REQUIRE(man.detach_by_id(id) == 0);
	REQUIRE(code.restored());
	REQUIRE(page_prot(code.entry) == (PROT_READ | PROT_EXEC));
	REQUIRE(page_prot(code.entry + 2) == (PROT_READ | PROT_WRITE | PROT_EXEC));
}
