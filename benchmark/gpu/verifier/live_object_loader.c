// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t exiting = 0;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static void sig_handler(int sig)
{
	(void)sig;
	exiting = 1;
}

static double elapsed_ms(const struct timespec *start,
			 const struct timespec *end)
{
	return (end->tv_sec - start->tv_sec) * 1000.0 +
	       (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

int main(int argc, char **argv)
{
	struct bpf_object *obj = NULL;
	struct bpf_program *prog = NULL;
	struct bpf_link *links[16] = {};
	size_t link_count = 0;
	int sleep_seconds = 8;
	int ret = 1;
	struct timespec load_begin = {};
	struct timespec load_end = {};
	struct timespec attach_end = {};

	if (argc < 2 || argc > 3) {
		fprintf(stderr, "usage: %s <bpf-object> [sleep-seconds]\n",
			argv[0]);
		return 1;
	}
	if (argc == 3) {
		sleep_seconds = atoi(argv[2]);
		if (sleep_seconds < 0) {
			fprintf(stderr, "invalid sleep seconds: %s\n", argv[2]);
			return 1;
		}
	}

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	obj = bpf_object__open_file(argv[1], NULL);
	if (!obj) {
		fprintf(stderr, "failed to open %s\n", argv[1]);
		goto cleanup;
	}

	clock_gettime(CLOCK_MONOTONIC, &load_begin);
	if (bpf_object__load(obj)) {
		fprintf(stderr, "failed to load %s: %s\n", argv[1],
			strerror(errno));
		goto cleanup;
	}
	clock_gettime(CLOCK_MONOTONIC, &load_end);

	bpf_object__for_each_program(prog, obj)
	{
		struct bpf_link *link;

		if (link_count >= sizeof(links) / sizeof(links[0])) {
			fprintf(stderr, "too many programs in %s\n", argv[1]);
			goto cleanup;
		}
		link = bpf_program__attach(prog);
		if (!link) {
			fprintf(stderr, "failed to attach program %s\n",
				bpf_program__name(prog));
			goto cleanup;
		}
		links[link_count++] = link;
	}
	clock_gettime(CLOCK_MONOTONIC, &attach_end);

	printf("loaded=%s programs=%zu load_ms=%.3f attach_ms=%.3f total_ms=%.3f pid=%d\n",
	       argv[1], link_count, elapsed_ms(&load_begin, &load_end),
	       elapsed_ms(&load_end, &attach_end),
	       elapsed_ms(&load_begin, &attach_end), getpid());
	fflush(stdout);

	for (int i = 0; i < sleep_seconds && !exiting; ++i) {
		sleep(1);
	}
	ret = 0;

cleanup:
	for (size_t i = 0; i < link_count; ++i) {
		bpf_link__destroy(links[i]);
	}
	if (obj) {
		bpf_object__close(obj);
	}
	return ret;
}
