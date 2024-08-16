#include "ebpf-vm.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <sys/cdefs.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <ebpf-vm.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <errno.h>
#include <stdbool.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

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
#define PT_REGS_PARM1(x) ((x)->di)
#define PT_REGS_PARM2(x) ((x)->si)
#define PT_REGS_PARM3(x) ((x)->dx)

#elif defined(__aarch64__) || defined(_M_ARM64)
struct pt_regs {
	uint64_t regs[31];
	uint64_t sp;
	uint64_t pc;
	uint64_t pstate;
};
#define PT_REGS_PARM1(x) ((x)->regs[0])
#define PT_REGS_PARM2(x) ((x)->regs[1])
#define PT_REGS_PARM3(x) ((x)->regs[2])
#else
#error "Unsupported architecture"
#endif

struct ebpf_vm *begin_vm = NULL;
struct ebpf_vm *end_vm = NULL;

bool enable_ebpf = false;

// The timespec struct holds seconds and nanoseconds
struct timespec start_time, end_time;

void start_timer()
{
	clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
}

void end_timer()
{
	clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
}

__attribute_noinline__ uint64_t __bench_probe(const char *a, int b,
							   uint64_t c)
{
	return a[b] + c;
}

uint64_t test_func_wrapper(const char *a, int b, uint64_t c)
{
	struct pt_regs regs;
	uint64_t ret;
	if (enable_ebpf) {
		memset(&regs, 0, sizeof(regs));
		PT_REGS_PARM1(&regs) = (uintptr_t)a;
		PT_REGS_PARM2(&regs) = b;
		PT_REGS_PARM3(&regs) = c;
		ebpf_exec(begin_vm, &regs, sizeof(regs), &ret);
	}
	uint64_t hook_func_ret = __bench_probe(a, b, c);
	if (enable_ebpf) {
		memset(&regs, 0, sizeof(regs));
		PT_REGS_PARM1(&regs) = hook_func_ret;
		ebpf_exec(end_vm, &regs, sizeof(regs), &ret);
	}
	return hook_func_ret;
}

static double get_elapsed_time()
{
	long seconds = end_time.tv_sec - start_time.tv_sec;
	long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
	if (start_time.tv_nsec > end_time.tv_nsec) { // clock underflow
		--seconds;
		nanoseconds += 1000000000;
	}
	printf("Elapsed time: %ld.%09ld seconds\n", seconds, nanoseconds);
	return seconds * 1.0 + nanoseconds / 1000000000.0;
}

static double get_function_time(int iter)
{
	start_timer();
	// test base line
	for (int i = 0; i < iter; i++) {
		test_func_wrapper("hello", i % 4, i);
	}
	end_timer();
	double time = get_elapsed_time();
	return time;
}

void do_benchmark_userspace(int iter)
{
	double base_line_time, after_hook_time, total_time;

	printf("a[b] + c for %d times\n", iter);
	base_line_time = get_function_time(iter);
	printf("avg function elapse time: %lf ns\n\n",
	       (base_line_time) / iter * 1000000000.0);
}

struct ebpf_vm *create_vm_from_elf(const char *elf_file,
				   ebpf_jit_fn *compiled_fn)
{
	LIBBPF_OPTS(bpf_object_open_opts, open_opts);
	int err;
	struct ebpf_vm *vm = NULL;
	struct bpf_object *obj = bpf_object__open_file(elf_file, &open_opts);
	if (!obj) {
		fprintf(stderr, "Failed to open elf file, errno=%d\n", errno);
		return NULL;
	}
	struct bpf_program *prog = bpf_object__next_program(obj, NULL);
	if (!prog) {
		fprintf(stderr, "No program found in %s\n", elf_file);
		goto out;
	}
	vm = ebpf_create();
	if (!vm) {
		goto out;
	}
	char *errmsg;
	err = ebpf_load(vm, bpf_program__insns(prog),
			bpf_program__insn_cnt(prog) * 8, &errmsg);
	if (err != 0) {
		fprintf(stderr, "Failed to load program: %s\n", errmsg);
		free(errmsg);
		goto out;
	}
	if (compiled_fn) {
		*compiled_fn = ebpf_compile(vm, &errmsg);
		if (!*compiled_fn) {
			fprintf(stderr, "Failed to compile: %s\n", errmsg);
			free(errmsg);
			goto err_out;
		}
	}
	goto out;
err_out:
	if (vm) {
		ebpf_destroy(vm);
	}

out:
	if (obj)
		bpf_object__close(obj);
	return vm;
}

int main(int argc,  char **argv)
{
	if (argc < 3) {
		printf("Usage: %s <uprobe elf> <uretprobe elf>\n", argv[0]);
		return 0;
	}
	const char *uprobe_prog = argv[1];
	const char *uretprobe_prog = argv[2];
	printf("uprobe elf: %s\nuretprobe elf:%s\n", uprobe_prog,
	       uretprobe_prog);
	enable_ebpf = true;
	begin_vm = create_vm_from_elf(uprobe_prog, NULL);
	assert(begin_vm);
	end_vm = create_vm_from_elf(uretprobe_prog, NULL);
	assert(end_vm);
	int iter = 100 * 1000;
	do_benchmark_userspace(iter);
	ebpf_destroy(begin_vm);
	ebpf_destroy(end_vm);
	return 0;
}
