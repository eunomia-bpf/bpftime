// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <gelf.h>
#include "./.output/launchlate.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

#define DEFAULT_UPROBE_SYMBOL_HINT "cudaLaunchKernel"

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

static Elf *open_elf(const char *path, int *fd_close)
{
	int fd;
	Elf *e;

	if (elf_version(EV_CURRENT) == EV_NONE) {
		warn("elf init failed\n");
		return NULL;
	}

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		warn("Could not open %s: %s\n", path, strerror(errno));
		return NULL;
	}

	e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e) {
		warn("elf_begin failed for %s: %s\n", path, elf_errmsg(-1));
		close(fd);
		return NULL;
	}

	if (elf_kind(e) != ELF_K_ELF) {
		warn("%s is not an ELF file\n", path);
		elf_end(e);
		close(fd);
		return NULL;
	}

	*fd_close = fd;
	return e;
}

static void close_elf(Elf *e, int fd_close)
{
	if (e)
		elf_end(e);
	if (fd_close >= 0)
		close(fd_close);
}

static char *find_defined_symbol_containing(const char *path, const char *needle)
{
	Elf *e = NULL;
	Elf_Scn *scn = NULL;
	Elf_Data *data = NULL;
	GElf_Shdr shdr;
	GElf_Sym sym;
	int fd = -1;

	e = open_elf(path, &fd);
	if (!e)
		return NULL;

	while ((scn = elf_nextscn(e, scn))) {
		if (!gelf_getshdr(scn, &shdr))
			continue;
		if (!(shdr.sh_type == SHT_SYMTAB || shdr.sh_type == SHT_DYNSYM))
			continue;

		data = NULL;
		while ((data = elf_getdata(scn, data))) {
			int i;

			for (i = 0; gelf_getsym(data, i, &sym); i++) {
				const char *name;

				if (sym.st_shndx == SHN_UNDEF)
					continue;
				if (GELF_ST_TYPE(sym.st_info) != STT_FUNC)
					continue;

				name = elf_strptr(e, shdr.sh_link, sym.st_name);
				if (!name)
					continue;
				if (!strstr(name, needle))
					continue;

				name = strdup(name);
				close_elf(e, fd);
				return (char *)name;
			}
		}
	}

	close_elf(e, fd);
	return NULL;
}

static int print_histogram(struct launchlate_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	uint32_t i;
	uint64_t value;
	int err = 0;
	int fd = bpf_map__fd(obj->maps.time_histogram);
	uint64_t total = 0;

	// Time range labels for each bin
	const char *labels[] = {
		"0-100ns",
		"100ns-1us",
		"1-10us",
		"10-100us",
		"100us-1ms",
		"1-10ms",
		"10-100ms",
		"100ms-1s",
		"1s-10s",
		">10s"
	};

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	printf("\n%-9s Launch Latency Distribution:\n", ts);
	printf("%-15s : count    distribution\n", "latency");

	// Read all histogram bins
	for (i = 0; i < 10; i++) {
		err = bpf_map_lookup_elem(fd, &i, &value);
		if (err && errno != ENOENT) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		if (!err && value > 0) {
			total += value;
		}
	}

	// Print histogram
	for (i = 0; i < 10; i++) {
		value = 0;
		err = bpf_map_lookup_elem(fd, &i, &value);
		if (err && errno != ENOENT) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}

		if (value > 0) {
			printf("%-15s : %-8" PRIu64 " |", labels[i], value);

			// Print histogram bar
			int bar_len = (value * 40) / (total > 0 ? total : 1);
			if (bar_len == 0 && value > 0)
				bar_len = 1;
			for (int j = 0; j < bar_len; j++)
				printf("*");
			printf("\n");
		}
	}

	printf("Total samples: %" PRIu64 "\n", total);
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	struct launchlate_bpf *skel;
	int err;
	struct timespec ts_mono, ts_real;
	int64_t offset_ns;
	uint32_t key = 0;
	const char *binary_path = "./vec_add";
	char *func_name = NULL;

	if (argc > 1)
		binary_path = argv[1];

	func_name = find_defined_symbol_containing(binary_path,
						   DEFAULT_UPROBE_SYMBOL_HINT);
	if (!func_name) {
		fprintf(stderr,
			"Failed to find a defined symbol containing '%s' in %s\n",
			DEFAULT_UPROBE_SYMBOL_HINT, binary_path);
		return 1;
	}

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = launchlate_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		free(func_name);
		return 1;
	}

	/* Load & verify BPF programs */
	err = launchlate_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	/* Calibrate clocks: compute offset between CLOCK_REALTIME and CLOCK_MONOTONIC */
	if (clock_gettime(CLOCK_MONOTONIC, &ts_mono) < 0) {
		fprintf(stderr, "Failed to get CLOCK_MONOTONIC: %s\n", strerror(errno));
		goto cleanup;
	}
	if (clock_gettime(CLOCK_REALTIME, &ts_real) < 0) {
		fprintf(stderr, "Failed to get CLOCK_REALTIME: %s\n", strerror(errno));
		goto cleanup;
	}

	/* Calculate offset: realtime - monotonic */
	offset_ns = (int64_t)(ts_real.tv_sec * 1000000000ULL + ts_real.tv_nsec) -
		    (int64_t)(ts_mono.tv_sec * 1000000000ULL + ts_mono.tv_nsec);

	printf("Clock calibration: REALTIME - MONOTONIC = %ld ns\n", offset_ns);
	printf("  MONOTONIC: %ld.%09ld\n", ts_mono.tv_sec, ts_mono.tv_nsec);
	printf("  REALTIME:  %ld.%09ld\n", ts_real.tv_sec, ts_real.tv_nsec);

	/* Store offset in BPF map */
	err = bpf_map_update_elem(bpf_map__fd(skel->maps.clock_offset), &key, &offset_ns, BPF_ANY);
	if (err) {
		fprintf(stderr, "Failed to update clock_offset map: %s\n", strerror(errno));
		goto cleanup;
	}

	printf("Attaching uprobe: binary_path='%s', func_name='%s' (auto-resolved from ELF)\n",
	       binary_path, func_name);

	/* Manually attach uprobe with configurable name */
	LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
		.func_name = func_name,
		.retprobe = false,
	);

	skel->links.uprobe_cuda_launch = bpf_program__attach_uprobe_opts(
		skel->progs.uprobe_cuda_launch, -1, binary_path, 0, &uprobe_opts);
	if (!skel->links.uprobe_cuda_launch) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe to '%s:%s': %s\n",
			binary_path, func_name, strerror(errno));
		goto cleanup;
	}

	/* Attach kprobe */
	err = launchlate_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF kprobe\n");
		goto cleanup;
	}

	printf("\nMonitoring CUDA kernel launch latency (uprobe: %s:%s)... Hit Ctrl-C to end.\n",
	       binary_path, func_name);

	while (!exiting) {
		sleep(2);  // Update every 2 seconds
		print_histogram(skel);
	}

cleanup:
	free(func_name);
	/* Clean up */
	launchlate_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
