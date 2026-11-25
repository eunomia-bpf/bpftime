// References the trampoline entry points that its inline assembly exports.
//
// With `BPFTIME_ENABLE_LTO=ON` the labels exist solely inside an `__asm__`
// block, so link-time optimization removes the function that emits them and
// these references come out as undefined symbols at link time (issue #83).
// The check therefore fails to link, which is the point.
//
// Only the exported labels are named here. `syscall_hooker_cxx` is
// deliberately *not*: the emitted code reaches it with `call
// syscall_hooker_cxx`, which LTO cannot see, so naming it here would keep it
// alive on its own and hide the second layer of the bug. This way the first
// symbol fails while the block is dropped, and `syscall_hooker_cxx` fails
// while the block survives without it.
#include <cstdint>
#include <cstdio>

#if defined(__x86_64__)
extern "C" void syscall_hooker_asm();
extern "C" int64_t call_orig_syscall(int64_t sys_nr, int64_t arg1,
				     int64_t arg2, int64_t arg3, int64_t arg4,
				     int64_t arg5, int64_t arg6);

int main()
{
	volatile uintptr_t addrs[] = { (uintptr_t)&syscall_hooker_asm,
				       (uintptr_t)&call_orig_syscall };
	for (uintptr_t addr : addrs) {
		if (addr == 0) {
			fprintf(stderr, "a trampoline symbol resolved to null\n");
			return 1;
		}
	}
	printf("syscall_hooker_asm=%p call_orig_syscall=%p\n", (void *)addrs[0],
	       (void *)addrs[1]);
	return 0;
}
#else
// The trampoline exists only for x86_64; nothing to check elsewhere.
int main()
{
	return 0;
}
#endif