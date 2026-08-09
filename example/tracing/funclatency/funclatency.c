// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2021 Google LLC.
 *
 * Based on funclatency from BCC by Brendan Gregg and others
 * 2021-02-26   Barret Rhoden   Created this.
 *
 * TODO:
 * - support uprobes on libraries without -p PID. (parse ld.so.cache)
 * - support regexp pattern matching and per-function histograms
 */
#define _GNU_SOURCE
#include <argp.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <bpf/btf.h>
#include "funclatency.h"
#include "funclatency.skel.h"
#include <stdbool.h>
#include <gelf.h>

#define warn(...) fprintf(stderr, __VA_ARGS__)

#define min(x, y)                                                              \
	({                                                                     \
		typeof(x) _min1 = (x);                                         \
		typeof(y) _min2 = (y);                                         \
		(void)(&_min1 == &_min2);                                      \
		_min1 < _min2 ? _min1 : _min2;                                 \
	})

/*
 * Opens an elf at `path` of kind ELF_K_ELF.  Returns NULL on failure.  On
 * success, close with close_elf(e, fd_close).
 */
Elf *open_elf(const char *path, int *fd_close)
{
	int fd;
	Elf *e;

	if (elf_version(EV_CURRENT) == EV_NONE) {
		warn("elf init failed\n");
		return NULL;
	}
	fd = open(path, O_RDONLY);
	if (fd < 0) {
		warn("Could not open %s\n", path);
		return NULL;
	}
	e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e) {
		warn("elf_begin failed: %s\n", elf_errmsg(-1));
		close(fd);
		return NULL;
	}
	if (elf_kind(e) != ELF_K_ELF) {
		warn("elf kind %d is not ELF_K_ELF\n", elf_kind(e));
		elf_end(e);
		close(fd);
		return NULL;
	}
	*fd_close = fd;
	return e;
}

void close_elf(Elf *e, int fd_close)
{
	elf_end(e);
	close(fd_close);
}

/* Returns the offset of a function in the elf file `path`, or -1 on failure. */
off_t get_elf_func_offset(const char *path, const char *func)
{
	off_t ret = -1;
	int i, fd = -1;
	Elf *e;
	Elf_Scn *scn;
	Elf_Data *data;
	GElf_Ehdr ehdr;
	GElf_Shdr shdr[1];
	GElf_Phdr phdr;
	GElf_Sym sym[1];
	size_t shstrndx, nhdrs;
	char *n;

	e = open_elf(path, &fd);

	if (!gelf_getehdr(e, &ehdr))
		goto out;

	if (elf_getshdrstrndx(e, &shstrndx) != 0)
		goto out;

	scn = NULL;
	while ((scn = elf_nextscn(e, scn))) {
		if (!gelf_getshdr(scn, shdr))
			continue;
		if (!(shdr->sh_type == SHT_SYMTAB ||
		      shdr->sh_type == SHT_DYNSYM))
			continue;
		data = NULL;
		while ((data = elf_getdata(scn, data))) {
			for (i = 0; gelf_getsym(data, i, sym); i++) {
				n = elf_strptr(e, shdr->sh_link, sym->st_name);
				if (!n)
					continue;
				if (GELF_ST_TYPE(sym->st_info) != STT_FUNC)
					continue;
				if (!strcmp(n, func)) {
					ret = sym->st_value;
					goto check;
				}
			}
		}
	}

check:
	if (ehdr.e_type == ET_EXEC || ehdr.e_type == ET_DYN) {
		if (elf_getphdrnum(e, &nhdrs) != 0) {
			ret = -1;
			goto out;
		}
		for (i = 0; i < (int)nhdrs; i++) {
			if (!gelf_getphdr(e, i, &phdr))
				continue;
			if (phdr.p_type != PT_LOAD || !(phdr.p_flags & PF_X))
				continue;
			if (phdr.p_vaddr <= ret &&
			    ret < (phdr.p_vaddr + phdr.p_memsz)) {
				ret = ret - phdr.p_vaddr + phdr.p_offset;
				goto out;
			}
		}
		ret = -1;
	}
out:
	close_elf(e, fd);
	return ret;
}

/*
 * Returns 0 on success; -1 on failure.  On sucess, returns via `path` the full
 * path to the program for pid.
 */
int get_pid_binary_path(pid_t pid, char *path, size_t path_sz)
{
	ssize_t ret;
	char proc_pid_exe[32];

	if (snprintf(proc_pid_exe, sizeof(proc_pid_exe), "/proc/%d/exe", pid) >=
	    sizeof(proc_pid_exe)) {
		warn("snprintf /proc/PID/exe failed");
		return -1;
	}
	ret = readlink(proc_pid_exe, path, path_sz);
	if (ret < 0) {
		warn("No such pid %d\n", pid);
		return -1;
	}
	if (ret >= path_sz) {
		warn("readlink truncation");
		return -1;
	}
	path[ret] = '\0';

	return 0;
}

/*
 * Returns 0 on success; -1 on failure.  On success, returns via `path` the full
 * path to a library matching the name `lib` that is loaded into pid's address
 * space.
 */
int get_pid_lib_path(pid_t pid, const char *lib, char *path, size_t path_sz)
{
	FILE *maps;
	char *p;
	char proc_pid_maps[32];
	char line_buf[1024];
	char path_buf[1024];

	if (snprintf(proc_pid_maps, sizeof(proc_pid_maps), "/proc/%d/maps",
		     pid) >= sizeof(proc_pid_maps)) {
		warn("snprintf /proc/PID/maps failed");
		return -1;
	}
	maps = fopen(proc_pid_maps, "r");
	if (!maps) {
		warn("No such pid %d\n", pid);
		return -1;
	}
	while (fgets(line_buf, sizeof(line_buf), maps)) {
		if (sscanf(line_buf, "%*x-%*x %*s %*x %*s %*u %s", path_buf) !=
		    1)
			continue;
		/* e.g. /usr/lib/x86_64-linux-gnu/libc-2.31.so */
		p = strrchr(path_buf, '/');
		if (!p)
			continue;
		if (strncmp(p, "/lib", 4))
			continue;
		p += 4;
		if (strncmp(lib, p, strlen(lib)))
			continue;
		p += strlen(lib);
		/* libraries can have - or . after the name */
		if (*p != '.' && *p != '-')
			continue;
		if (strnlen(path_buf, 1024) >= path_sz) {
			warn("path size too small\n");
			return -1;
		}
		strcpy(path, path_buf);
		fclose(maps);
		return 0;
	}

	warn("Cannot find library %s\n", lib);
	fclose(maps);
	return -1;
}

/*
 * Returns 0 on success; -1 on failure.  On success, returns via `path` the full
 * path to the program.
 */
static int which_program(const char *prog, char *path, size_t path_sz)
{
	FILE *which;
	char cmd[100];

	if (snprintf(cmd, sizeof(cmd), "which %s", prog) >= sizeof(cmd)) {
		warn("snprintf which prog failed");
		return -1;
	}
	which = popen(cmd, "r");
	if (!which) {
		warn("which failed");
		return -1;
	}
	if (!fgets(path, path_sz, which)) {
		warn("fgets which failed");
		pclose(which);
		return -1;
	}
	/* which has a \n at the end of the string */
	path[strlen(path) - 1] = '\0';
	pclose(which);
	return 0;
}

/*
 * Returns 0 on success; -1 on failure.  On success, returns via `path` the full
 * path to the binary for the given pid.
 * 1) pid == x, binary == ""    : returns the path to x's program
 * 2) pid == x, binary == "foo" : returns the path to libfoo linked in x
 * 3) pid == 0, binary == ""    : failure: need a pid or a binary
 * 4) pid == 0, binary == "bar" : returns the path to `which bar`
 *
 * For case 4), ideally we'd like to search for libbar too, but we don't support
 * that yet.
 */
int resolve_binary_path(const char *binary, pid_t pid, char *path,
			size_t path_sz)
{
	if (!strcmp(binary, "")) {
		if (!pid) {
			warn("Uprobes need a pid or a binary\n");
			return -1;
		}
		return get_pid_binary_path(pid, path, path_sz);
	}
	if (pid)
		return get_pid_lib_path(pid, binary, path, path_sz);

	if (which_program(binary, path, path_sz)) {
		/*
		 * If the user is tracing a program by name, we can find it.
		 * But we can't find a library by name yet.  We'd need to parse
		 * ld.so.cache or something similar.
		 */
		warn("Can't find %s (Need a PID if this is a library)\n",
		     binary);
		return -1;
	}
	return 0;
}


static void print_stars(unsigned int val, unsigned int val_max, int width)
{
	int num_stars, num_spaces, i;
	bool need_plus;

	num_stars = min(val, val_max) * width / val_max;
	num_spaces = width - num_stars;
	need_plus = val > val_max;

	for (i = 0; i < num_stars; i++)
		printf("*");
	for (i = 0; i < num_spaces; i++)
		printf(" ");
	if (need_plus)
		printf("+");
}

static void print_log2_hist(unsigned int *vals, int vals_size,
			    const char *val_type)
{
	int stars_max = 40, idx_max = -1;
	unsigned int val, val_max = 0;
	unsigned long long low, high;
	int stars, width, i;

	for (i = 0; i < vals_size; i++) {
		val = vals[i];
		if (val > 0)
			idx_max = i;
		if (val > val_max)
			val_max = val;
	}

	if (idx_max < 0)
		return;

	printf("%*s%-*s : count    distribution\n", idx_max <= 32 ? 5 : 15, "",
	       idx_max <= 32 ? 19 : 29, val_type);

	if (idx_max <= 32)
		stars = stars_max;
	else
		stars = stars_max / 2;

	for (i = 0; i <= idx_max; i++) {
		low = (1ULL << (i + 1)) >> 1;
		high = (1ULL << (i + 1)) - 1;
		if (low == high)
			low -= 1;
		val = vals[i];
		width = idx_max <= 32 ? 10 : 20;
		printf("%*lld -> %-*lld : %-8d |", width, low, width, high,
		       val);
		print_stars(val, val_max, stars);
		printf("|\n");
	}
}

#define warn(...) fprintf(stderr, __VA_ARGS__)

static struct prog_env {
	int units;
	pid_t pid;
	unsigned int duration;
	unsigned int interval;
	unsigned int iterations;
	bool timestamp;
	char *funcname;
	bool verbose;
} env = {
	.interval = 99999999,
	.iterations = 99999999,
};

const char *argp_program_version = "funclatency 0.1";
const char *argp_program_bug_address =
	"https://github.com/iovisor/bcc/tree/master/libbpf-tools";
static const char args_doc[] = "FUNCTION";
static const char program_doc[] =
	"Time functions and print latency as a histogram\n"
	"\n"
	"Usage: funclatency [-h] [-m|-u] [-p PID] [-d DURATION] [ -i INTERVAL ] [-c CG]\n"
	"                   [-T] FUNCTION\n"
	"       Choices for FUNCTION: FUNCTION         (kprobe)\n"
	"                             LIBRARY:FUNCTION (uprobe a library in -p PID)\n"
	"                             :FUNCTION        (uprobe the binary of -p PID)\n"
	"                             PROGRAM:FUNCTION (uprobe the binary PROGRAM)\n"
	"\v"
	"Examples:\n"
	"  ./funclatency do_sys_open         # time the do_sys_open() kernel function\n"
	"  ./funclatency -m do_nanosleep     # time do_nanosleep(), in milliseconds\n"
	"  ./funclatency -c CG               # Trace process under cgroupsPath CG\n"
	"  ./funclatency -u vfs_read         # time vfs_read(), in microseconds\n"
	"  ./funclatency -p 181 vfs_read     # time process 181 only\n"
	"  ./funclatency -p 181 c:read       # time the read() C library function\n"
	"  ./funclatency -p 181 :foo         # time foo() from pid 181's userspace\n"
	"  ./funclatency -i 2 -d 10 vfs_read # output every 2 seconds, for 10s\n"
	"  ./funclatency -mTi 5 vfs_read     # output every 5 seconds, with timestamps\n";

static const struct argp_option opts[] = {
	{ "milliseconds", 'm', NULL, 0, "Output in milliseconds" },
	{ "microseconds", 'u', NULL, 0, "Output in microseconds" },
	{ 0, 0, 0, 0, "" },
	{ "pid", 'p', "PID", 0, "Process ID to trace" },
	{ 0, 0, 0, 0, "" },
	{ "interval", 'i', "INTERVAL", 0, "Summary interval in seconds" },
	{ "duration", 'd', "DURATION", 0, "Duration to trace" },
	{ "timestamp", 'T', NULL, 0, "Print timestamp" },
	{ "verbose", 'v', NULL, 0, "Verbose debug output" },
	{ NULL, 'h', NULL, OPTION_HIDDEN, "Show the full help" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	struct prog_env *env = state->input;
	long duration, interval, pid;

	switch (key) {
	case 'p':
		errno = 0;
		pid = strtol(arg, NULL, 10);
		if (errno || pid <= 0) {
			warn("Invalid PID: %s\n", arg);
			argp_usage(state);
		}
		env->pid = pid;
		break;
	case 'm':
		if (env->units != NSEC) {
			warn("only set one of -m or -u\n");
			argp_usage(state);
		}
		env->units = MSEC;
		break;
	case 'u':
		if (env->units != NSEC) {
			warn("only set one of -m or -u\n");
			argp_usage(state);
		}
		env->units = USEC;
		break;
	case 'd':
		errno = 0;
		duration = strtol(arg, NULL, 10);
		if (errno || duration <= 0) {
			warn("Invalid duration: %s\n", arg);
			argp_usage(state);
		}
		env->duration = duration;
		break;
	case 'i':
		errno = 0;
		interval = strtol(arg, NULL, 10);
		if (errno || interval <= 0) {
			warn("Invalid interval: %s\n", arg);
			argp_usage(state);
		}
		env->interval = interval;
		break;
	case 'T':
		env->timestamp = true;
		break;
	case 'v':
		env->verbose = true;
		break;
	case 'h':
		argp_state_help(state, stderr, ARGP_HELP_STD_HELP);
		break;
	case ARGP_KEY_ARG:
		if (env->funcname) {
			warn("Too many function names: %s\n", arg);
			argp_usage(state);
		}
		env->funcname = arg;
		break;
	case ARGP_KEY_END:
		if (!env->funcname) {
			warn("Need a function to trace\n");
			argp_usage(state);
		}
		if (env->duration) {
			if (env->interval > env->duration)
				env->interval = env->duration;
			env->iterations = env->duration / env->interval;
		}
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static const char *unit_str(void)
{
	switch (env.units) {
	case NSEC:
		return "nsec";
	case USEC:
		return "usec";
	case MSEC:
		return "msec";
	};

	return "bad units";
}


static int attach_uprobes(struct funclatency_bpf *obj)
{
	char *binary, *function;
	char bin_path[PATH_MAX];
	off_t func_off;
	int ret = -1;
	long err;

	binary = strdup(env.funcname);
	if (!binary) {
		warn("strdup failed");
		return -1;
	}
	function = strchr(binary, ':');
	if (!function) {
		warn("Binary should have contained ':' (internal bug!)\n");
		return -1;
	}
	*function = '\0';
	function++;

	if (resolve_binary_path(binary, env.pid, bin_path, sizeof(bin_path)))
		goto out_binary;

	func_off = get_elf_func_offset(bin_path, function);
	if (func_off < 0) {
		warn("Could not find %s in %s\n", function, bin_path);
		goto out_binary;
	}

	obj->links.dummy_kprobe =
		bpf_program__attach_uprobe(obj->progs.dummy_kprobe, false,
					   env.pid ?: -1, bin_path, func_off);
	if (!obj->links.dummy_kprobe) {
		err = -errno;
		warn("Failed to attach uprobe: %ld\n", err);
		goto out_binary;
	}

	obj->links.dummy_kretprobe =
		bpf_program__attach_uprobe(obj->progs.dummy_kretprobe, true,
					   env.pid ?: -1, bin_path, func_off);
	if (!obj->links.dummy_kretprobe) {
		err = -errno;
		warn("Failed to attach uretprobe: %ld\n", err);
		goto out_binary;
	}

	ret = 0;

out_binary:
	free(binary);

	return ret;
}

static volatile bool exiting;

static void sig_hand(int signr)
{
	exiting = true;
}

static struct sigaction sigact = { .sa_handler = sig_hand };

int main(int argc, char **argv)
{
	LIBBPF_OPTS(bpf_object_open_opts, open_opts);
	static const struct argp argp = {
		.options = opts,
		.parser = parse_arg,
		.args_doc = args_doc,
		.doc = program_doc,
	};
	struct funclatency_bpf *obj;
	int i, err;
	struct tm *tm;
	char ts[32];
	time_t t;
	bool used_fentry = false;

	err = argp_parse(&argp, argc, argv, 0, NULL, &env);
	if (err)
		return err;

	sigaction(SIGINT, &sigact, 0);

	libbpf_set_print(libbpf_print_fn);

	obj = funclatency_bpf__open_opts(&open_opts);
	if (!obj) {
		warn("failed to open BPF object\n");
		return 1;
	}

	obj->rodata->units = env.units;
	obj->rodata->targ_tgid = env.pid;


	err = funclatency_bpf__load(obj);
	if (err) {
		warn("failed to load BPF object\n");
		return 1;
	}

	if (!obj->bss) {
		warn("Memory-mapping BPF maps is supported starting from Linux 5.7, please upgrade.\n");
		goto cleanup;
	}

	if (!used_fentry) {
		err = attach_uprobes(obj);
		if (err)
			goto cleanup;
	}

	err = funclatency_bpf__attach(obj);
	if (err) {
		fprintf(stderr, "failed to attach BPF programs: %s\n",
			strerror(-err));
		goto cleanup;
	}

	printf("Tracing %s.  Hit Ctrl-C to exit\n", env.funcname);

	for (i = 0; i < env.iterations && !exiting; i++) {
		sleep(env.interval);

		printf("\n");
		if (env.timestamp) {
			time(&t);
			tm = localtime(&t);
			strftime(ts, sizeof(ts), "%H:%M:%S", tm);
			printf("%-8s\n", ts);
		}

		print_log2_hist(obj->bss->hist, MAX_SLOTS, unit_str());

		/* Cleanup histograms for interval output */
		memset(obj->bss->hist, 0, sizeof(obj->bss->hist));
	}

	printf("Exiting trace of %s\n", env.funcname);

cleanup:
	funclatency_bpf__destroy(obj);

	return err != 0;
}
