#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <vector>
#include <cstdint>
#include <unistd.h>
#include <string>
#include <cinttypes>
#include "text_segment_transformer.hpp"
#include <frida-gum.h>
/*
Function arguments are passed using the following order:
- RDI
- RSI
- RDX
- RCX
- R8
- R9
- ...stack

- RAX: return value

While syscall args are passed using
- RAX:  syscall_nr
- RDI:  arg1
- RSI:  arg2
- RDX:  arg3
- R10:  arg4
- R8:   arg5
- R9:   arg6

- RAX: return value

*/

extern "C" void syscall_hooker_asm();
extern "C" int64_t call_orig_syscall(int64_t sys_nr, int64_t arg1, int64_t arg2,
				     int64_t arg3, int64_t arg4, int64_t arg5,
				     int64_t arg6);
extern "C" void syscall_addr();

static const int NR_syscalls = 512;
static syscall_hooker_func_t call_hook = &call_orig_syscall;

#if defined(__x86_64__)
[[maybe_unused]] void __asm_holder()
{
	__asm__(".globl syscall_hooker_asm\n\t"
		"syscall_hooker_asm:\n\t"
		"pop %rax\n\t" // Restore the saved rax, which is the syscall
			       // number
		"cmp $15, %rax\n\t" // Special handing for rt_sigreturn, they
				    // don't need to be traced
		"je handle_sigreturn\n\t"
		"movq (%rsp), %rcx\n\t"
		"pushq %rbp\n\t"
		"movq %rsp, %rbp\n\t"
		"andq $-16, %rsp \n\t" // 16 byte stack alignment
		"pushq %rax\n\t" // Save syscall args on stack
		"pushq %rdi\n\t"
		"pushq %rsi\n\t"
		"pushq %rdx\n\t"
		"pushq %r10\n\t"
		"pushq %r8\n\t"
		"pushq %r9\n\t"
		// Put syscall args in appreciate argument order
		"movq 48(%rsp), %rdi\n\t" // syscall_nr
		"movq 40(%rsp), %rsi\n\t" // arg1
		"movq 32(%rsp), %rdx\n\t" // arg2
		"movq 24(%rsp), %rcx\n\t" // arg3
		"movq 16(%rsp), %r8\n\t" // arg4
		"movq 8(%rsp), %r9\n\t" // arg5
		"pushq (%rsp)\n\t" // arg6
		"call syscall_hooker_cxx\n\t"
		"leave\n\t"
		"ret\n\t");

	__asm__(".globl call_orig_syscall\n\t"
		"call_orig_syscall:\n\t"
		"movq %rdi, %rax \n\t"
		"movq %rsi, %rdi \n\t"
		"movq %rdx, %rsi \n\t"
		"movq %rcx, %rdx \n\t"
		"movq %r8, %r10 \n\t"
		"movq %r9, %r8 \n\t"
		"movq 8(%rsp),%r9 \n\t"
		"handle_sigreturn:\n\t"
		// "addq $8, %rsp\n\t"
		"syscall_addr:\n\t"
		"syscall\n\t"
		"ret\n\t");
}
#elif defined(__aarch64__)
// TODO: implement syscall trace trampoline
#else
#error "Unsupported architecture"
#endif

extern "C" int64_t syscall_hooker_cxx(int64_t sys_nr, int64_t arg1,
				      int64_t arg2, int64_t arg3, int64_t arg4,
				      int64_t arg5, int64_t arg6)
{
	return call_hook(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
}

static inline void rewrite_segment(uint8_t *code, size_t len, int perm)
{
	// Set the pages to be writable
	if (int err = mprotect(code, len, PROT_READ | PROT_WRITE | PROT_EXEC);
	    err < 0) {
		SPDLOG_ERROR(
			"Failed to change the protect status of the rewriting page {:x}",
			(uintptr_t)code);
		exit(1);
	}
	csh cs_handle;
	cs_err ret;
	ret = cs_open(CS_ARCH_X86, CS_MODE_64, &cs_handle);
	if (ret != CS_ERR_OK) {
		SPDLOG_ERROR("Failed to open capstone instance: {}, {}",
			      (int)ret, cs_strerror(ret));
		exit(1);
	}
	const uint8_t *curr_code = code;
	size_t size = len;
	uint64_t curr_addr = (uint64_t)(uintptr_t)curr_code;
	cs_insn curr_insn;
	memset(&curr_insn, 0, sizeof(curr_insn));
	while (curr_addr < (uintptr_t)code + len) {
		auto ok = cs_disasm_iter(cs_handle, &curr_code, &size,
					 &curr_addr, &curr_insn);
		if (!ok) {
			break;
		}
		auto insn_name =
			std::string(cs_insn_name(cs_handle, curr_insn.id));
		if (insn_name == "syscall" || insn_name == "sysenter") {
			if (curr_insn.address != (uintptr_t)&syscall_addr) {
				uint8_t *curr_pos =
					(uint8_t *)(uintptr_t)curr_insn.address;
				SPDLOG_TRACE("Rewrite syscall insn at {}",
					      (void *)curr_pos);
				curr_pos[0] = 0xff;
				curr_pos[1] = 0xd0;
			}
		}
	}
	cs_close(&cs_handle);
	if (int err = mprotect(code, len, perm); err < 0) {
		SPDLOG_ERROR(
			"Failed to change the protect status of the rewriting page {:x}",
			(uintptr_t)code);
		exit(1);
	}
}

struct MapEntry {
	uint64_t begin, end;
	char w, r, x;
	int get_perm() const
	{
		int ret = 0;
		if (w == 'w')
			ret |= PROT_WRITE;
		if (r == 'r')
			ret |= PROT_READ;
		if (x == 'x')
			ret |= PROT_EXEC;
		return ret;
	}
};
namespace bpftime
{

syscall_hooker_func_t get_call_hook()
{
	return call_hook;
}
void set_call_hook(syscall_hooker_func_t hook)
{
	call_hook = hook;
}

void setup_syscall_tracer()
{
	// Setup page mappings

	if (auto mmap_addr =
		    mmap(0x0, 0x1000, PROT_EXEC | PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, -1, 0);
	    mmap_addr == MAP_FAILED) {
		SPDLOG_ERROR("Failed to perform mmap: errno={}, message={}",
			      errno, strerror(errno));
		exit(1);
	}
	// Setup jumpings
	for (int i = 0; i < NR_syscalls; i++) {
		// 0x90; nop
		*((char *)(uintptr_t)(i)) = 0x90;
	}
	// Jump to the syscall handler function after the nop-s
	/*
	50
	push %rax;

	48 b8 88 77 66 55 44 33 22 11
	movabs $0x1122334455667788, %rax; // The constant is the address
	of syscall_hooker_asm

	ff e0
	jmp *%rax;

	*/
	std::vector<uint8_t> codes;
	codes.push_back(0x50);
	codes.push_back(0x48);
	codes.push_back(0xb8);
	for (int i = 0; i < 8; i++) {
		codes.push_back(
			(uint8_t)((((uint64_t)(uintptr_t)syscall_hooker_asm) >>
				   (8 * i)) &
				  0xff));
	}
	codes.push_back(0xff);
	codes.push_back(0xe0);
	std::copy(codes.begin(), codes.end(),
		  (uint8_t *)(uintptr_t)(0 + NR_syscalls));
	// Set the page to execute-only. Keep normal behavior of
	// dereferencing null-pointers
	if (int err = mprotect(0, 0x1000, PROT_EXEC); err < 0) {
		SPDLOG_ERROR(
			"Failed to set execute only of 0-started page: {}",
			errno);
		exit(1);
	}

	SPDLOG_INFO("Page zero setted up..");
	// Scan for /proc/self/maps

	std::vector<MapEntry> entries;
	std::ifstream ifs("/proc/self/maps");
	while (ifs) {
		std::string line;
		std::getline(ifs, line);

		MapEntry curr;
		char *path_buf;
		int cnt = sscanf(line.c_str(),
				 "%" SCNx64 "-%" SCNx64
				 " %c%c%c%*c %*x %*x:%*x %*d %ms",
				 &curr.begin, &curr.end, &curr.r, &curr.w,
				 &curr.x, &path_buf);
		if (cnt < 5)
			continue;
		if (cnt == 6) {
			std::string buf = path_buf;
			free(path_buf);
			if (buf == "[stack]" || buf == "[vsyscall]") {
				continue;
			}
		}

		entries.push_back(curr);
	}
	SPDLOG_INFO("Rewriting executable segments..");
	// Hack the executable mappings
	for (const auto &map : entries) {
		if (map.x == 'x') {
			if (map.begin == 0) {
				// Skip pages that we mapped
				continue;
			}
			SPDLOG_DEBUG("Rewriting segment from {:x} to {:x}",
				      map.begin, map.end);
			rewrite_segment((uint8_t *)(uintptr_t)(map.begin),
					map.end - map.begin, map.get_perm());
		}
	}
}

} // namespace bpftime
