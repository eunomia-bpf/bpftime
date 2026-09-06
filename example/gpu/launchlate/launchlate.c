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
#include <limits.h>
#include <gelf.h>
#include "./.output/launchlate.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

#define DEFAULT_UPROBE_SYMBOL_HINT "_Z9vectorAddPKfS0_Pf"
#define CUDA_LAUNCH_SYMBOL "cudaLaunchKernel"

enum symbol_match_status {
	SYMBOL_ABSENT = 0,
	SYMBOL_UNDEFINED = 1,
	SYMBOL_NOT_FUNCTION = 2,
	SYMBOL_INVALID_VALUE = 3,
	SYMBOL_AMBIGUOUS = 4,
};

struct host_target {
	uint64_t launch_vaddr;
	uint64_t kernel_vaddr;
	uint64_t valid;
};

struct plt_entry {
	uint64_t file_offset;
	uint64_t vaddr;
};

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

static int find_defined_symbol_matching(const char *path, const char *needle,
					enum symbol_match_status *status,
					uint64_t *vaddr)
{
	Elf *e = NULL;
	Elf_Scn *scn = NULL;
	Elf_Data *data = NULL;
	GElf_Shdr shdr;
	GElf_Sym sym;
	bool found = false;
	int fd = -1;

	*status = SYMBOL_ABSENT;
	*vaddr = 0;
	e = open_elf(path, &fd);
	if (!e)
		return -EINVAL;

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

				name = elf_strptr(e, shdr.sh_link, sym.st_name);
				if (!name)
					continue;
				if (strcmp(name, needle) != 0)
					continue;
				if (sym.st_shndx == SHN_UNDEF) {
					if (*status == SYMBOL_ABSENT)
						*status = SYMBOL_UNDEFINED;
					continue;
				}
				if (GELF_ST_TYPE(sym.st_info) != STT_FUNC) {
					*status = SYMBOL_NOT_FUNCTION;
					continue;
				}
				Elf_Scn *function_scn = elf_getscn(e, sym.st_shndx);
				GElf_Shdr function_shdr;
				if (!sym.st_value || !function_scn ||
				    !gelf_getshdr(function_scn, &function_shdr) ||
				    !(function_shdr.sh_flags & SHF_EXECINSTR) ||
				    sym.st_value < function_shdr.sh_addr ||
				    sym.st_value - function_shdr.sh_addr >=
					    function_shdr.sh_size) {
					*status = SYMBOL_INVALID_VALUE;
					continue;
				}
				if (found && *vaddr != sym.st_value) {
					*status = SYMBOL_AMBIGUOUS;
					close_elf(e, fd);
					return -EEXIST;
				}
				*vaddr = sym.st_value;
				found = true;
			}
		}
	}

	close_elf(e, fd);
	return found ? 0 : -ENOENT;
}

static int find_x86_64_plt_entry(const char *path, const char *needle,
				 struct plt_entry *entry)
{
	Elf *e = NULL;
	Elf_Scn *scn = NULL, *rel_scn = NULL;
	Elf_Data *rel_data, *sym_data;
	GElf_Ehdr ehdr;
	GElf_Shdr shdr, rel_shdr = {0}, sym_shdr, plt_shdr = {0};
	GElf_Rela rela;
	GElf_Sym sym;
	size_t shstrndx, matched_index = 0, relocation_count;
	uint64_t entry_index, entry_offset;
	bool have_plt_sec = false, matched = false;
	int fd = -1, result = -ENOENT;

	memset(entry, 0, sizeof(*entry));
	e = open_elf(path, &fd);
	if (!e)
		return -EINVAL;
	if (!gelf_getehdr(e, &ehdr) || elf_getshdrstrndx(e, &shstrndx) != 0 ||
	    ehdr.e_machine != EM_X86_64) {
		result = -ENOTSUP;
		goto out;
	}

	while ((scn = elf_nextscn(e, scn))) {
		const char *name;

		if (!gelf_getshdr(scn, &shdr))
			continue;
		name = elf_strptr(e, shstrndx, shdr.sh_name);
		if (!name)
			continue;
		if (strcmp(name, ".plt.sec") == 0) {
			plt_shdr = shdr;
			have_plt_sec = true;
		} else if (strcmp(name, ".plt") == 0 && !have_plt_sec) {
			plt_shdr = shdr;
		} else if (strcmp(name, ".rela.plt") == 0) {
			if (rel_scn) {
				result = -EINVAL;
				goto out;
			}
			rel_scn = scn;
			rel_shdr = shdr;
		}
	}
	if (!rel_scn || !plt_shdr.sh_size ||
	    plt_shdr.sh_type != SHT_PROGBITS ||
	    !(plt_shdr.sh_flags & SHF_EXECINSTR) || plt_shdr.sh_entsize != 16 ||
	    rel_shdr.sh_type != SHT_RELA || !rel_shdr.sh_entsize) {
		result = -EINVAL;
		goto out;
	}
	{
		Elf_Scn *sym_scn = elf_getscn(e, rel_shdr.sh_link);
		if (!sym_scn || !gelf_getshdr(sym_scn, &sym_shdr) ||
		    !sym_shdr.sh_entsize) {
			result = -EINVAL;
			goto out;
		}
		sym_data = elf_getdata(sym_scn, NULL);
		if (!sym_data || sym_data->d_size != sym_shdr.sh_size ||
		    elf_getdata(sym_scn, sym_data)) {
			result = -EINVAL;
			goto out;
		}
	}

	rel_data = elf_getdata(rel_scn, NULL);
	if (!rel_data || rel_data->d_size != rel_shdr.sh_size ||
	    elf_getdata(rel_scn, rel_data)) {
		result = -EINVAL;
		goto out;
	}
	relocation_count = rel_data->d_size / rel_shdr.sh_entsize;
	for (size_t i = 0; i < relocation_count; i++) {
		const char *name;
		if (!gelf_getrela(rel_data, i, &rela) ||
		    !gelf_getsym(sym_data, GELF_R_SYM(rela.r_info), &sym))
			continue;
		name = elf_strptr(e, sym_shdr.sh_link, sym.st_name);
		if (!name || strcmp(name, needle) != 0)
			continue;
		if (matched) {
			result = -EEXIST;
			goto out;
		}
		matched = true;
		matched_index = i;
	}
	if (!matched)
		goto out;

	entry_index = matched_index;
	if (!have_plt_sec) {
		if (entry_index == UINT64_MAX) {
			result = -ERANGE;
			goto out;
		}
		entry_index++;
	}
	if (entry_index > UINT64_MAX / 16) {
		result = -ERANGE;
		goto out;
	}
	entry_offset = entry_index * 16;
	if (entry_offset > plt_shdr.sh_size ||
	    plt_shdr.sh_entsize > plt_shdr.sh_size - entry_offset ||
	    entry_offset > UINT64_MAX - plt_shdr.sh_offset ||
	    entry_offset > UINT64_MAX - plt_shdr.sh_addr) {
		result = -ERANGE;
		goto out;
	}
	entry->file_offset = plt_shdr.sh_offset + entry_offset;
	entry->vaddr = plt_shdr.sh_addr + entry_offset;
	result = 0;

out:
	close_elf(e, fd);
	return result;
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
	int queue_fd = bpf_map__fd(obj->maps.queue_state);
	uint64_t queue_values[4] = {0};
	for (i = 0; i < 4; i++) {
		uint32_t queue_key = i;
		if (bpf_map_lookup_elem(queue_fd, &queue_key, &queue_values[i]) != 0) {
			warn("queue_state lookup failed: %s\n", strerror(errno));
			return -1;
		}
	}
	printf("Host launches: %" PRIu64 "\n", queue_values[0]);
	printf("Device entries: %" PRIu64 "\n", queue_values[1]);
	printf("Queue underflows: %" PRIu64 "\n", queue_values[2]);
	printf("Queue overflows: %" PRIu64 "\n", queue_values[3]);
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	struct launchlate_bpf *skel;
	struct host_target target = {0};
	struct plt_entry launch_plt = {0};
	int err;
	struct timespec ts_mono, ts_real;
	int64_t offset_ns;
	uint32_t key = 0;
	const char *binary_path = "./vec_add";
	const char *symbol_hint = DEFAULT_UPROBE_SYMBOL_HINT;
	enum symbol_match_status symbol_status;

	if (argc > 1)
		binary_path = argv[1];
	if (argc > 2)
		symbol_hint = argv[2];

	err = find_defined_symbol_matching(binary_path, symbol_hint, &symbol_status,
					   &target.kernel_vaddr);
	if (err) {
		fprintf(stderr,
			"Failed to find a defined symbol exactly matching '%s' in %s\n",
			symbol_hint, binary_path);
		return 1;
	}

	err = find_x86_64_plt_entry(binary_path, CUDA_LAUNCH_SYMBOL,
				    &launch_plt);
	if (err || launch_plt.file_offset > SIZE_MAX) {
		fprintf(stderr, "Failed to find the '%s' PLT entry in %s\n",
			CUDA_LAUNCH_SYMBOL, binary_path);
		return 1;
	}
	target.launch_vaddr = launch_plt.vaddr;
	target.valid = 1;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = launchlate_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = launchlate_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	/* Publish the launch PLT and target virtual addresses for the host filter */
	err = bpf_map_update_elem(bpf_map__fd(skel->maps.host_target), &key,
				  &target, BPF_ANY);
	if (err) {
		err = -errno;
		fprintf(stderr, "Failed to update host_target map: %s\n",
			strerror(errno));
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

	printf("Attaching uprobe: binary_path='%s', target='%s', launch='%s@plt'\n",
	       binary_path, symbol_hint, CUDA_LAUNCH_SYMBOL);

	/* Attach at the target ELF's launch PLT entry; arg0 is filtered in BPF. */
	LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
		.retprobe = false,
	);

	skel->links.uprobe_cuda_launch = bpf_program__attach_uprobe_opts(
		skel->progs.uprobe_cuda_launch, -1, binary_path,
		(size_t)launch_plt.file_offset, &uprobe_opts);
	if (!skel->links.uprobe_cuda_launch) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe to '%s:%s@plt': %s\n",
			binary_path, CUDA_LAUNCH_SYMBOL, strerror(errno));
		goto cleanup;
	}

	/* Attach kprobe */
	err = launchlate_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF kprobe\n");
		goto cleanup;
	}

	printf("\nMonitoring CUDA kernel launch latency (uprobe: %s:%s@plt, target: %s)... Hit Ctrl-C to end.\n",
	       binary_path, CUDA_LAUNCH_SYMBOL, symbol_hint);

	while (!exiting)
		sleep(1);
	print_histogram(skel);

cleanup:
	/* Clean up */
	launchlate_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
