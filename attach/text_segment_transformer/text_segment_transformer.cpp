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
#include <sys/syscall.h>
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
				     int64_t arg6, int64_t user_ip,
				     int64_t user_sp, int64_t user_bp);
extern "C" void syscall_addr();

static const int NR_syscalls = 512;
static const size_t LOW_TRAMPOLINE_SLOT_SIZE = 16;
static uintptr_t low_trampoline_begin;
static uintptr_t low_trampoline_end;
static bool use_low_trampoline;
static syscall_hooker_func_t call_hook = &call_orig_syscall;
extern "C" {
thread_local uintptr_t bpftime_vfork_return_ip = 0;
thread_local unsigned bpftime_syscall_dispatch_active = 0;
}

#if defined(__x86_64__)
[[maybe_unused]] void __asm_holder()
{
	__asm__(".globl syscall_hooker_asm\n\t"
		"syscall_hooker_asm:\n\t"
		"pop %rax\n\t" // Restore the saved rax, which is the syscall
			       // number
		"cmp $15, %rax\n\t" // Special handing for rt_sigreturn, they
					    // don't need to be traced
		"je handle_rt_sigreturn\n\t"
		// clone and clone3 return TWICE. The child resumes at the `ret`
		// of call_orig_syscall, but on its own fresh stack, where no
		// return address was ever pushed -- it pops whatever the thread
		// stack happens to hold and jumps there. They therefore cannot
		// be traced through the C hook at all; send them down the
		// untraced path, after seeding the child stack so that the
		// child's `ret` lands where the replaced `syscall` instruction
		// would have continued.
		"cmp $56, %rax\n\t" // SYS_clone
		"je handle_clone\n\t"
		"cmp $435, %rax\n\t" // SYS_clone3
		"je handle_clone3\n\t"
		// fork and vfork also return in two processes. In particular, the
		// vfork child shares the parent's stack until exec or _exit, so it
		// must never run through the C++ dispatcher on that stack.
		"cmp $57, %rax\n\t" // SYS_fork
		"je syscall_addr\n\t"
		"cmp $58, %rax\n\t" // SYS_vfork
		"je handle_vfork\n\t"
		// The dispatcher and its helpers may issue syscalls themselves. Sending
		// those through the dispatcher again recursively corrupts its private
		// stack, so keep nested syscalls on the raw path until this callback
		// returns.
		"movq bpftime_syscall_dispatch_active@gottpoff(%rip), %rcx\n\t"
		"cmpl $0, %fs:(%rcx)\n\t"
		"jne syscall_addr\n\t"
		"movl $1, %fs:(%rcx)\n\t"
		// Retain the application stack pointer for pt_regs and the final ret.
		"movq %rsp, %r11\n\t"
		"pushq %rbp\n\t" // application frame pointer
		"pushq %r11\n\t" // application stack with return PC at top
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
		// The OTel stack helper consumes a pt_regs-shaped context. Carry
		// the application state at the replaced syscall through the C ABI:
		// arg6, return PC, pre-hook SP, and the original frame pointer.
		"subq $8, %rsp\n\t" // preserve 16-byte call-site alignment
		"pushq 8(%rbp)\n\t" // user_bp (argument 10)
		"movq 0(%rbp), %r11\n\t"
		"leaq 8(%r11), %r11\n\t"
		"pushq %r11\n\t" // user_sp (argument 9)
		"movq 0(%rbp), %r11\n\t"
		"pushq (%r11)\n\t" // user_ip (argument 8)
		"pushq 32(%rsp)\n\t" // arg6 (argument 7)
		"call syscall_hooker_cxx\n\t"
		"movq bpftime_syscall_dispatch_active@gottpoff(%rip), %rcx\n\t"
		"movl $0, %fs:(%rcx)\n\t"
		// A real syscall preserves the argument registers. The C++
		// dispatcher is allowed to clobber them, so restore their values
		// before returning to the application while retaining the syscall
		// result in %rax.
		"movq %rax, %r11\n\t"
		"movq 80(%rsp), %rdi\n\t"
		"movq 72(%rsp), %rsi\n\t"
		"movq 64(%rsp), %rdx\n\t"
		"movq 56(%rsp), %r10\n\t"
		"movq 48(%rsp), %r8\n\t"
		"movq 40(%rsp), %r9\n\t"
		"movq %r11, %rax\n\t"
		"movq 0(%rbp), %rcx\n\t"
		"movq 8(%rbp), %rbp\n\t"
		"movq %rcx, %rsp\n\t"
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
		"syscall_addr:\n\t"
		"syscall\n\t"
		"ret\n\t"
		// Replacing rt_sigreturn's two-byte syscall with call *%rax adds
		// one return address above the kernel signal frame. Remove it before
		// asking the kernel to restore that frame. This syscall cannot return
		// on success.
		"handle_rt_sigreturn:\n\t"
		"addq $8, %rsp\n\t"
		"syscall\n\t"
		"ud2\n\t"
		// The vfork child shares the parent's stack. Remove the return slot
		// added by the replacement call before entering the kernel, and keep
		// the continuation in TLS so neither return path leaves a bpftime
		// frame for the child to overwrite.
		"handle_vfork:\n\t"
		"movq (%rsp), %r11\n\t"
		"movq bpftime_vfork_return_ip@gottpoff(%rip), %rcx\n\t"
		"movq %r11, %fs:(%rcx)\n\t"
		"addq $8, %rsp\n\t"
		"syscall\n\t"
		"movq bpftime_vfork_return_ip@gottpoff(%rip), %rcx\n\t"
		"jmp *%fs:(%rcx)\n\t"
		// Reached only by an explicit jump from the dispatcher, never
		// by fall-through: call_orig_syscall above must continue
		// straight into `syscall`. Registers are still in syscall ABI
		// order and the application's return address is at (%rsp),
		// because only the pushed syscall number has been popped.
		// %rcx and %r11 are free scratch: the `syscall` destroys both.
		"handle_clone:\n\t"
		"testq %rsi, %rsi\n\t" // no child stack: fork-like, the child
					// keeps the parent's stack layout and
					// the ordinary `ret` is already correct
		"jz syscall_addr\n\t"
		"movq (%rsp), %r11\n\t" // application return address
		"subq $8, %rsi\n\t"
		"movq %r11, (%rsi)\n\t" // the child pops this in `ret`
		"jmp syscall_addr\n\t"
		"handle_clone3:\n\t"
		"movq 40(%rdi), %rcx\n\t" // clone_args.stack
		"testq %rcx, %rcx\n\t"
		"jz syscall_addr\n\t"
		"cmpq $8, 48(%rdi)\n\t" // clone_args.stack_size
		"jb syscall_addr\n\t"
		"addq 48(%rdi), %rcx\n\t" // top of the child stack
		"movq (%rsp), %r11\n\t"
		"subq $8, %rcx\n\t"
		"movq %r11, (%rcx)\n\t"
		"subq $8, 48(%rdi)\n\t" // the kernel starts the child one
					 // slot lower, on the seeded address
		"jmp syscall_addr\n\t");
}
#elif defined(__aarch64__)
// TODO: implement syscall trace trampoline
#else
#error "Unsupported architecture"
#endif

extern "C" int64_t syscall_hooker_cxx(int64_t sys_nr, int64_t arg1,
				      int64_t arg2, int64_t arg3, int64_t arg4,
				      int64_t arg5, int64_t arg6,
				      int64_t user_ip, int64_t user_sp,
				      int64_t user_bp)
{
	return call_hook(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6, user_ip,
			 user_sp, user_bp);
}

static bool is_memory_syscall(int64_t sysno)
{
	return sysno == __NR_mmap || sysno == __NR_munmap ||
	       sysno == __NR_mremap || sysno == __NR_brk;
}

static bool is_accumulator_register(unsigned reg)
{
	return reg == X86_REG_RAX || reg == X86_REG_EAX || reg == X86_REG_AX ||
	       reg == X86_REG_AL || reg == X86_REG_AH;
}

static bool setup_low_syscall_trampoline()
{
	constexpr size_t slots_size = NR_syscalls * LOW_TRAMPOLINE_SLOT_SIZE;
	constexpr size_t mapping_size = (slots_size + 12 + 0xfff) & ~0xfff;
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
	void *mapping = MAP_FAILED;
	for (uintptr_t address = 0x10000; address < 0x10000000;
	     address += 0x10000) {
		mapping = mmap((void *)address, mapping_size,
			       PROT_READ | PROT_WRITE,
			       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
			       -1, 0);
		if (mapping != MAP_FAILED)
			break;
	}
	if (mapping == MAP_FAILED)
		return false;

	auto *base = static_cast<uint8_t *>(mapping);
	auto *common = base + slots_size;
	for (int sysno = 0; sysno < NR_syscalls; ++sysno) {
		auto *slot = base + sysno * LOW_TRAMPOLINE_SLOT_SIZE;
		memset(slot, 0x90, LOW_TRAMPOLINE_SLOT_SIZE);
		slot[0] = 0xb8; // mov $sysno, %eax
		memcpy(slot + 1, &sysno, sizeof(uint32_t));
		slot[5] = 0x50; // push %rax
		slot[6] = 0xe9; // jmp common
		int32_t relative = (int32_t)(common - (slot + 11));
		memcpy(slot + 7, &relative, sizeof(relative));
	}
	common[0] = 0x48;
	common[1] = 0xb8; // movabs $syscall_hooker_asm, %rax
	uintptr_t handler = (uintptr_t)syscall_hooker_asm;
	memcpy(common + 2, &handler, sizeof(handler));
	common[10] = 0xff;
	common[11] = 0xe0; // jmp *%rax
	if (mprotect(mapping, mapping_size, PROT_READ | PROT_EXEC) != 0) {
		int saved_errno = errno;
		munmap(mapping, mapping_size);
		errno = saved_errno;
		return false;
	}
	low_trampoline_begin = (uintptr_t)mapping;
	low_trampoline_end = low_trampoline_begin + mapping_size;
	return true;
}

static bool rewrite_segment(uint8_t *code, size_t len, int perm)
{
	// Set the pages to be writable
	if (int err = mprotect(code, len, PROT_READ | PROT_WRITE | PROT_EXEC);
	    err < 0) {
		SPDLOG_ERROR(
			"Failed to change the protect status of the rewriting page {:x}",
			(uintptr_t)code);
		return false;
	}
	csh cs_handle;
	cs_err ret;
	ret = cs_open(CS_ARCH_X86, CS_MODE_64, &cs_handle);
	if (ret != CS_ERR_OK) {
		SPDLOG_ERROR("Failed to open capstone instance: {}, {}",
			      (int)ret, cs_strerror(ret));
		(void)mprotect(code, len, perm);
		return false;
	}
	if (use_low_trampoline)
		cs_option(cs_handle, CS_OPT_DETAIL, CS_OPT_ON);
	struct constant_syscall_candidate {
		uint8_t *immediate = nullptr;
		uint8_t size = 0;
		int64_t sysno = -1;
	} candidate;
	const uint8_t *curr_code = code;
	size_t size = len;
	uint64_t curr_addr = (uint64_t)(uintptr_t)curr_code;
	cs_insn *curr_insn = cs_malloc(cs_handle);
	if (curr_insn == nullptr) {
		SPDLOG_ERROR("Failed to allocate capstone instruction");
		cs_close(&cs_handle);
		(void)mprotect(code, len, perm);
		return false;
	}
	while (curr_addr < (uintptr_t)code + len) {
		auto ok = cs_disasm_iter(cs_handle, &curr_code, &size,
					 &curr_addr, curr_insn);
		if (!ok) {
			break;
		}
		auto insn_name =
			std::string(cs_insn_name(cs_handle, curr_insn->id));
		if (insn_name == "syscall" || insn_name == "sysenter") {
			if (curr_insn->address != (uintptr_t)&syscall_addr) {
				uint8_t *curr_pos =
					(uint8_t *)(uintptr_t)curr_insn->address;
				if (use_low_trampoline) {
					if (candidate.immediate == nullptr ||
					    !is_memory_syscall(candidate.sysno)) {
						candidate = {};
						continue;
					}
					uint64_t target = low_trampoline_begin +
						candidate.sysno * LOW_TRAMPOLINE_SLOT_SIZE;
					if ((candidate.size == 4 && target > UINT32_MAX) ||
					    (candidate.size != 4 && candidate.size != 8)) {
						candidate = {};
						continue;
					}
					memcpy(candidate.immediate, &target,
					       candidate.size);
				}
				SPDLOG_TRACE("Rewrite syscall insn at {}",
					      (void *)curr_pos);
				curr_pos[0] = 0xff;
				curr_pos[1] = 0xd0;
			}
			candidate = {};
			continue;
		}
		if (!use_low_trampoline || curr_insn->detail == nullptr)
			continue;

		cs_regs regs_read = {};
		cs_regs regs_write = {};
		uint8_t read_count = 0;
		uint8_t write_count = 0;
		bool touches_accumulator =
			cs_regs_access(cs_handle, curr_insn, regs_read, &read_count,
				       regs_write, &write_count) != CS_ERR_OK;
		for (uint8_t i = 0; !touches_accumulator && i < read_count; ++i)
			touches_accumulator = is_accumulator_register(regs_read[i]);
		for (uint8_t i = 0; !touches_accumulator && i < write_count; ++i)
			touches_accumulator = is_accumulator_register(regs_write[i]);
		if (touches_accumulator ||
		    cs_insn_group(cs_handle, curr_insn, CS_GRP_JUMP) ||
		    cs_insn_group(cs_handle, curr_insn, CS_GRP_CALL) ||
		    cs_insn_group(cs_handle, curr_insn, CS_GRP_RET) ||
		    cs_insn_group(cs_handle, curr_insn, CS_GRP_INT) ||
		    cs_insn_group(cs_handle, curr_insn, CS_GRP_IRET))
			candidate = {};

		const auto &x86 = curr_insn->detail->x86;
		if (curr_insn->id == X86_INS_MOV && x86.op_count >= 2 &&
		    x86.operands[0].type == X86_OP_REG &&
		    is_accumulator_register(x86.operands[0].reg) &&
		    x86.operands[1].type == X86_OP_IMM &&
		    (x86.encoding.imm_size == 4 || x86.encoding.imm_size == 8)) {
			candidate.immediate =
				(uint8_t *)(uintptr_t)curr_insn->address +
				x86.encoding.imm_offset;
			candidate.size = x86.encoding.imm_size;
			candidate.sysno = x86.operands[1].imm;
		}
	}
	cs_free(curr_insn, 1);
	cs_close(&cs_handle);
	if (int err = mprotect(code, len, perm); err < 0) {
		SPDLOG_ERROR(
			"Failed to change the protect status of the rewriting page {:x}",
			(uintptr_t)code);
		return false;
	}
	return true;
}

struct MapEntry {
	uint64_t begin, end;
	char w, r, x;
	std::string path;
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
		if (!setup_low_syscall_trampoline()) {
			SPDLOG_ERROR(
				"Failed to map either syscall trampoline: errno={}, message={}",
				errno, strerror(errno));
			return;
		}
		use_low_trampoline = true;
		SPDLOG_INFO("Using rootless low syscall trampoline at {:x}",
			    low_trampoline_begin);
	} else {
		// Setup jumpings
		auto *page_zero = static_cast<uint8_t *>(mmap_addr);
		for (int i = 0; i < NR_syscalls; i++) {
			// 0x90; nop
			page_zero[i] = 0x90;
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
	std::copy(codes.begin(), codes.end(), page_zero + NR_syscalls);
		// Set the page to execute-only. Keep normal behavior of
		// dereferencing null-pointers
		if (int err = mprotect(0, 0x1000, PROT_EXEC); err < 0) {
			SPDLOG_ERROR(
				"Failed to set execute only of 0-started page: {}",
				errno);
			(void)munmap(nullptr, 0x1000);
			return;
		}

		SPDLOG_INFO("Page zero setted up..");
	}
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
			curr.path = std::move(buf);
		}

		entries.push_back(curr);
	}
	SPDLOG_INFO("Rewriting executable segments..");
	std::string agent_path;
	if (const char *agent_so = getenv("AGENT_SO"); agent_so != nullptr) {
		if (char *resolved = realpath(agent_so, nullptr); resolved != nullptr) {
			agent_path = resolved;
			free(resolved);
		} else {
			agent_path = agent_so;
		}
	}
	std::string transformer_path;
	Dl_info transformer_info = {};
	if (dladdr((void *)&setup_syscall_tracer, &transformer_info) != 0 &&
	    transformer_info.dli_fname != nullptr) {
		if (char *resolved = realpath(transformer_info.dli_fname, nullptr);
		    resolved != nullptr) {
			transformer_path = resolved;
			free(resolved);
		} else {
			transformer_path = transformer_info.dli_fname;
		}
	}
	// Hack the executable mappings
	for (const auto &map : entries) {
		if (map.x == 'x') {
			if (map.begin == 0 ||
			    (map.begin < low_trampoline_end &&
			     map.end > low_trampoline_begin)) {
				// Skip pages that we mapped
				continue;
			}
			if (!agent_path.empty() && !map.path.empty()) {
				std::string map_path = map.path;
				if (char *resolved = realpath(map.path.c_str(), nullptr);
				    resolved != nullptr) {
					map_path = resolved;
					free(resolved);
				}
				if (map_path == agent_path ||
				    map_path == transformer_path) {
					SPDLOG_DEBUG(
						"Skipping bpftime agent executable segment from {:x} to {:x}",
						map.begin, map.end);
					continue;
				}
			}
			SPDLOG_DEBUG("Rewriting segment from {:x} to {:x}",
				      map.begin, map.end);
			rewrite_segment((uint8_t *)(uintptr_t)(map.begin),
					map.end - map.begin, map.get_perm());
		}
	}
}

} // namespace bpftime
