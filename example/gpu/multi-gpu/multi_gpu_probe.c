// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/**
 * multi_gpu_probe.c - Userspace loader for multi-GPU eBPF probe
 *
 * Loads the eBPF probe program and periodically prints kernel call
 * counts and accumulated execution times from the maps.
 */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include "./.output/multi_gpu_probe.skel.h"
#include <inttypes.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static int print_stat(struct multi_gpu_probe_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	uint32_t key, *prev_key = NULL;
	uint64_t value;
	int err = 0;

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	printf("\n%-9s --- Multi-GPU Kernel Stats ---\n", ts);

	// Print call counts
	int fd = bpf_map__fd(obj->maps.call_count);
	prev_key = NULL;
	printf("  Call counts:\n");
	while (1) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT) {
				err = 0;
				break;
			}
			warn("bpf_map_get_next_key failed: %s\n",
			     strerror(errno));
			return err;
		}
		err = bpf_map_lookup_elem(fd, &key, &value);
		if (err) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		printf("    block=%-5" PRIu32 " calls: %" PRIu64 "\n", key,
		       value);
		prev_key = &key;
	}

	// Print accumulated execution times (first 8 blocks)
	fd = bpf_map__fd(obj->maps.total_time_ns);
	prev_key = NULL;
	printf("  Execution times (ns):\n");
	int count = 0;
	while (count < 8) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT) {
				err = 0;
				break;
			}
			break;
		}
		err = bpf_map_lookup_elem(fd, &key, &value);
		if (err)
			break;
		printf("    block=%-5" PRIu32 " total_ns: %" PRIu64 "\n", key,
		       value);
		prev_key = &key;
		count++;
	}

	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	struct multi_gpu_probe_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = multi_gpu_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = multi_gpu_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton\n");
		goto cleanup;
	}

	err = multi_gpu_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	printf("Multi-GPU probe loaded. Waiting for kernel events...\n");
	printf("(Run multi_gpu_vec_add with bpftime to see traces)\n\n");

	while (!exiting) {
		sleep(2);
		print_stat(skel);
	}

cleanup:
	multi_gpu_probe_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
