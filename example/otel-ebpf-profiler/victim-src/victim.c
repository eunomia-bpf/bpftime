// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
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

int main(void)
{
	unsigned long iterations = read_ulong_env("VICTIM_ITERATIONS", 0);
	unsigned long alloc_size = read_ulong_env("VICTIM_ALLOC_SIZE", 1024);
	unsigned long sleep_us = read_ulong_env("VICTIM_SLEEP_US", 100 * 1000);
	unsigned long print_every = read_ulong_env("VICTIM_PRINT_EVERY", 1);
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
	}
	return 0;
}
