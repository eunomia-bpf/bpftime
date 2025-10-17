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
#include <string.h>
#include "./.output/cuda_probe.skel.h"
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

static void print_usage(const char *prog)
{
	fprintf(stderr, "Usage: %s [MODE]\n", prog);
	fprintf(stderr, "MODE:\n");
	fprintf(stderr, "  empty              - Empty probe (baseline)\n");
	fprintf(stderr, "  entry              - Entry probe only\n");
	fprintf(stderr, "  exit               - Exit probe only\n");
	fprintf(stderr, "  both               - Entry + Exit probes\n");
	fprintf(stderr, "  ringbuf            - Ring buffer test\n");
	fprintf(stderr, "  globaltimer        - Global timer test\n");
	fprintf(stderr, "  array-update       - Array map update\n");
	fprintf(stderr, "  array-lookup       - Array map lookup\n");
	fprintf(stderr, "  hash-update        - Hash map update\n");
	fprintf(stderr, "  hash-lookup        - Hash map lookup\n");
	fprintf(stderr, "  hash-delete        - Hash map delete\n");
	fprintf(stderr, "  pergputd-array-lookup - Per-GPU-thread array map lookup\n");
	fprintf(stderr, "  memtrace           - Memory trace test\n");
	fprintf(stderr, "  gpu-array-update   - GPU array map update\n");
	fprintf(stderr, "  gpu-array-lookup   - GPU array map lookup\n");
	fprintf(stderr, "\nDefault: empty\n");
}

static int attach_probes(struct cuda_probe_bpf *skel, const char *mode)
{
	// Disable all programs by default
	bpf_program__set_autoload(skel->progs.cuda__probe, false);
	bpf_program__set_autoload(skel->progs.cuda__retprobe, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_entry, false);
	bpf_program__set_autoload(skel->progs.cuda__retprobe_exit, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_both_entry, false);
	bpf_program__set_autoload(skel->progs.cuda__retprobe_both_exit, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_ringbuf, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_globaltimer, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_array_update, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_array_lookup, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_hash_update, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_hash_lookup, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_hash_delete, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_pergputd_array_lookup, false);
	bpf_program__set_autoload(skel->progs.cuda__probe_memtrace, false);
	bpf_program__set_autoload(skel->progs.cuda__retprobe_gpu_array_update, false);
	bpf_program__set_autoload(skel->progs.cuda__retprobe_gpu_array_lookup, false);

	// Enable the requested mode
	if (strcmp(mode, "empty") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe, true);
		bpf_program__set_autoload(skel->progs.cuda__retprobe, true);
		fprintf(stderr, "Mode: Empty probe (baseline)\n");
	} else if (strcmp(mode, "entry") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_entry, true);
		fprintf(stderr, "Mode: Entry probe only\n");
	} else if (strcmp(mode, "exit") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__retprobe_exit, true);
		fprintf(stderr, "Mode: Exit probe only\n");
	} else if (strcmp(mode, "both") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_both_entry, true);
		bpf_program__set_autoload(skel->progs.cuda__retprobe_both_exit, true);
		fprintf(stderr, "Mode: Entry + Exit probes\n");
	} else if (strcmp(mode, "ringbuf") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_ringbuf, true);
		fprintf(stderr, "Mode: Ring buffer test\n");
	} else if (strcmp(mode, "globaltimer") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_globaltimer, true);
		fprintf(stderr, "Mode: Global timer test\n");
	} else if (strcmp(mode, "array-update") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_array_update, true);
		fprintf(stderr, "Mode: Array map update\n");
	} else if (strcmp(mode, "array-lookup") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_array_lookup, true);
		fprintf(stderr, "Mode: Array map lookup\n");
	} else if (strcmp(mode, "hash-update") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_hash_update, true);
		fprintf(stderr, "Mode: Hash map update\n");
	} else if (strcmp(mode, "hash-lookup") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_hash_lookup, true);
		fprintf(stderr, "Mode: Hash map lookup\n");
	} else if (strcmp(mode, "hash-delete") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_hash_delete, true);
		fprintf(stderr, "Mode: Hash map delete\n");
	} else if (strcmp(mode, "pergputd-array-lookup") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_pergputd_array_lookup, true);
		fprintf(stderr, "Mode: Per-GPU-thread array map lookup\n");
	} else if (strcmp(mode, "memtrace") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__probe_memtrace, true);
		fprintf(stderr, "Mode: Memory trace test\n");
	} else if (strcmp(mode, "gpu-array-update") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__retprobe_gpu_array_update, true);
		fprintf(stderr, "Mode: GPU array map update\n");
	} else if (strcmp(mode, "gpu-array-lookup") == 0) {
		bpf_program__set_autoload(skel->progs.cuda__retprobe_gpu_array_lookup, true);
		fprintf(stderr, "Mode: GPU array map lookup\n");
	} else {
		fprintf(stderr, "Unknown mode: %s\n", mode);
		return -1;
	}

	return 0;
}

int main(int argc, char **argv)
{
	struct cuda_probe_bpf *skel;
	int err;
	const char *mode = "empty";

	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
			print_usage(argv[0]);
			return 0;
		}
		mode = argv[1];
	}

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = cuda_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Configure which probes to attach */
	err = attach_probes(skel, mode);
	if (err) {
		print_usage(argv[0]);
		goto cleanup;
	}

	/* Load & verify BPF programs */
	err = cuda_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	err = cuda_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	fprintf(stderr, "BPF probes attached. Waiting for target program...\n");

	while (!exiting) {
		sleep(1);
	}

cleanup:
	/* Clean up */
	cuda_probe_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
