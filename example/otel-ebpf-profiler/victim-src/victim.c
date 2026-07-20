// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static unsigned long read_ulong_env(const char *name, unsigned long fallback)
{
	const char *value = getenv(name);
	char *end = NULL;
	unsigned long parsed;

	if (!value || !*value) {
		return fallback;
	}

	parsed = strtoul(value, &end, 10);
	if (!end || *end != '\0') {
		fprintf(stderr, "invalid %s=%s\n", name, value);
		exit(2);
	}
	return parsed;
}

static unsigned long monotonic_seconds(void)
{
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
		perror("clock_gettime");
		exit(2);
	}
	return (unsigned long)ts.tv_sec;
}

int main(void)
{
	unsigned long iterations = read_ulong_env("VICTIM_ITERATIONS", 0);
	unsigned long run_seconds = read_ulong_env("VICTIM_RUN_SECONDS", 0);
	unsigned long alloc_size = read_ulong_env("VICTIM_ALLOC_SIZE", 1024);
	unsigned long sleep_us = read_ulong_env("VICTIM_SLEEP_US", 100 * 1000);
	unsigned long print_every = read_ulong_env("VICTIM_PRINT_EVERY", 1);
	unsigned long start = run_seconds > 0 ? monotonic_seconds() : 0;
	unsigned long i = 0;

	for (;;) {
		void *ptr = malloc(alloc_size);
		if (!ptr) {
			return 1;
		}
		if (print_every > 0 && i % print_every == 0) {
			printf("malloc/free loop %lu\n", i);
			fflush(stdout);
		}
		if (sleep_us > 0) {
			usleep(sleep_us);
		}
		free(ptr);
		i++;
		if (iterations > 0 && i >= iterations) {
			break;
		}
		if (run_seconds > 0 && (i & 0xfff) == 0 &&
		    monotonic_seconds() - start >= run_seconds) {
			break;
		}
	}
	printf("iterations_completed=%lu\n", i);
	return 0;
}
