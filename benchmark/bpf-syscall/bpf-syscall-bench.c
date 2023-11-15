// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include "linux/filter.h"

int prog_fd;

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

__attribute_noinline__ uint64_t __benchmark_test_function1(const char *a, int b,
							   uint64_t c)
{
	return bpf_prog_test_run_opts(prog_fd, NULL);
}

typedef uint64_t (*benchmark_test_function_t)(const char *, int, uint64_t);

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

static double get_function_time(benchmark_test_function_t func, int iter)
{
	start_timer();
	// test base line
	for (int i = 0; i < iter; i++) {
		func("hello", i % 4, i);
	}
	end_timer();
	double time = get_elapsed_time();
	return time;
}

void do_benchmark_userspace(benchmark_test_function_t func, int iter)
{
	double base_line_time;

	printf("a[b] + c for %d times\n", iter);
	base_line_time = get_function_time(func, iter);
	printf("Average time usage %lf ns\n\n",
	       (base_line_time) / iter * 1000000000.0);
}

#define do_benchmark_func(func, iter)                                   \
	do {                                                                   \
		printf("Benchmarking %s\n", #func);                           \
		do_benchmark_userspace(func ,iter);	\
	} while (0)

int test_run_time()
{
	puts("");
	struct bpf_insn trival_prog_insns[] = {
		BPF_MOV64_IMM(BPF_REG_0, 0),
		BPF_EXIT_INSN(),
	};
	int prog_fd = bpf_prog_load(BPF_PROG_TYPE_SOCKET_FILTER, NULL, "GPL",
		      trival_prog_insns, 2, NULL);
	if (prog_fd < 0) {
		printf("Failed to load BPF program: %s\n", strerror(prog_fd));
		exit(1);
	}
	int iter = 100 * 1000;
	do_benchmark_func(__benchmark_test_function1, iter);
    return 0;
}

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

int main(int argc, char **argv)
{
	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	test_run_time();

	return 0;
}
