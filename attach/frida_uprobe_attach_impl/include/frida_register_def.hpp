#ifndef _BPFTIME_FRIDA_PT_REGS
#define _BPFTIME_FRIDA_PT_REGS
#include <cstdint>
namespace bpftime
{

#if defined(__x86_64__) || defined(_M_X64)

struct pt_regs {
	uint64_t r15;
	uint64_t r14;
	uint64_t r13;
	uint64_t r12;
	uint64_t bp;
	uint64_t bx;
	uint64_t r11;
	uint64_t r10;
	uint64_t r9;
	uint64_t r8;
	uint64_t ax;
	uint64_t cx;
	uint64_t dx;
	uint64_t si;
	uint64_t di;
	uint64_t orig_ax;
	uint64_t ip;
	uint64_t cs;
	uint64_t flags;
	uint64_t sp;
	uint64_t ss;
};
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L79
#define PT_REGS_PARM1(x) ((x)->di)
#define PT_REGS_PARM2(x) ((x)->si)
#define PT_REGS_PARM3(x) ((x)->dx)
#define PT_REGS_PARM4(x) ((x)->cx)
#define PT_REGS_PARM5(x) ((x)->r8)
#define PT_REGS_PARM6(x) ((x)->r9)
#define PT_REGS_RET(x) ((x)->sp)
#define PT_REGS_RC(x) ((x)->ax)

#elif defined(__aarch64__) || defined(_M_ARM64)
struct pt_regs {
	uint64_t regs[31];
	uint64_t sp;
	uint64_t pc;
	uint64_t pstate;
};
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L217
#define PT_REGS_PARM1(x) ((x)->regs[0])
#define PT_REGS_PARM2(x) ((x)->regs[1])
#define PT_REGS_PARM3(x) ((x)->regs[2])
#define PT_REGS_PARM4(x) ((x)->regs[3])
#define PT_REGS_PARM5(x) ((x)->regs[4])
#define PT_REGS_PARM6(x) ((x)->regs[5])
#define PT_REGS_PARM7(x) ((x)->regs[6])
#define PT_REGS_PARM8(x) ((x)->regs[7])
#define PT_REGS_RET(x) ((x)->regs[30])
#define PT_REGS_RC(x) ((x)->regs[0])

#elif defined(__arm__) || defined(_M_ARM)
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L192
struct pt_regs {
	uint32_t uregs[18];
};
#define PT_REGS_PARM1(x) ((x)->uregs[0])
#define PT_REGS_PARM2(x) ((x)->uregs[1])
#define PT_REGS_PARM3(x) ((x)->uregs[2])
#define PT_REGS_PARM4(x) ((x)->uregs[3])
#define PT_REGS_RET(x) ((x)->uregs[14])
#define PT_REGS_RC(x) ((x)->uregs[0])
#else
#error "Unsupported architecture"
#endif

} // namespace bpftime

#endif
