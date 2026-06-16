// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "./.output/bb_register_probe.skel.h"

struct bb_reg_snapshot {
	uint64_t hits;
	uint64_t last_r2;
	uint64_t last_rd8;
};

static volatile bool exiting = false;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static void sig_handler(int sig)
{
	exiting = true;
}

static void print_snapshot(struct bb_register_probe_bpf *skel)
{
	struct bb_reg_snapshot snap = {};
	uint32_t key = 0;
	int fd;

	fd = bpf_map__fd(skel->maps.bb_regs);
	if (bpf_map_lookup_elem(fd, &key, &snap) != 0) {
		fprintf(stderr, "failed to read bb_regs map\n");
		return;
	}

	printf("hits=%" PRIu64 " r2=0x%" PRIx64 " rd8=0x%" PRIx64
	       " expected_r2(tid0)=0xdeadbeef expected_rd8(tid0)=0xdeadbef0\n",
	       snap.hits, snap.last_r2, snap.last_rd8);
	fflush(stdout);
}

int main(void)
{
	struct bb_register_probe_bpf *skel;
	int err;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = bb_register_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "failed to open BPF skeleton\n");
		return 1;
	}

	err = bb_register_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	err = bb_register_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "failed to attach BPF skeleton: %d\n", err);
		goto cleanup;
	}

	while (!exiting) {
		sleep(1);
		print_snapshot(skel);
	}

cleanup:
	bb_register_probe_bpf__destroy(skel);
	return err != 0;
}
