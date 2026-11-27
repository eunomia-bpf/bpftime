/* SPDX-License-Identifier: MIT
 *
 * End-to-end victim workload: exports a noinline compute() function and
 * exercises libsystem_malloc malloc()/free().
 *
 *   bpftime_darwin_e2e_victim [iters] [loop]
 * "loop" runs until SIGTERM for the running-process attach scenario.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#define EXPORT __attribute__((noinline, visibility("default")))
#else
#define EXPORT __attribute__((noinline, noclone))
#endif

EXPORT long compute(long a, long b)
{
	long r = a * 3 + b;
	asm volatile("" : "+r"(r));
	return r;
}

int main(int argc, char **argv)
{
	int iters = (argc > 1) ? atoi(argv[1]) : 20;
	int forever = (argc > 2 && strcmp(argv[2], "loop") == 0);
	printf("pid=%d\n", (int)getpid());
	fflush(stdout);
	for (int i = 0; forever || i < iters; i++) {
		void *p = malloc(64 + (i % 3) * 16);
		long r = compute(100 + i, 7);
		if (p)
			free(p);
		(void)r;
		if (forever)
			usleep(20 * 1000);
		else if (i == iters - 1)
			break;
	}
	printf("done\n");
	fflush(stdout);
	return 0;
}
