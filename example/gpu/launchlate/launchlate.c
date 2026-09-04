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
#include "rm_ptimer_575.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

#define DEFAULT_UPROBE_SYMBOL_HINT "_Z9vectorAddPKfS0_Pf"
#define CUDA_LAUNCH_SYMBOL "cudaLaunchKernel"
#define HIST_BINS 10
#define LAUNCH_QUEUE_SIZE 4096
#define HOST_COUNTERS 6
#define DEVICE_COUNTERS 6
#define CALIBRATION_WARMUPS 4
#define CALIBRATION_TRIALS 32
#define CLOCK_DRIFT_LIMIT_PPB 10000ULL

enum host_counter {
	HOST_LAUNCHES = 0,
	HOST_ENQUEUED = 1,
	QUEUE_OVERFLOWS = 2,
	QUEUE_UPDATE_ERRORS = 3,
	HOST_LAUNCH_CALLS = 4,
	HOST_TARGET_ERRORS = 5,
};

enum device_counter {
	DEVICE_ENTRIES = 0,
	MATCHED_SAMPLES = 1,
	QUEUE_UNDERFLOWS = 2,
	CLASSIFIED_SAMPLES = 3,
	UNCERTAIN_SAMPLES = 4,
	CLOCK_ERRORS = 5,
};

struct clock_calibration {
	int64_t offset_low_ns;
	int64_t offset_high_ns;
	uint64_t uncertainty_ns;
	uint64_t host_anchor_ns;
	uint64_t valid;
};

struct clock_drift {
	int64_t offset_change_low_ns;
	int64_t offset_change_high_ns;
	uint64_t elapsed_ns;
	uint64_t rate_bound_ppb;
	int bounded;
};

struct rm_calibration_run {
	uint64_t requested;
	uint64_t accepted;
	uint64_t rejected;
	uint64_t cleanup_complete;
	struct rm_ptimer_575_sample best;
};

struct launch_sample {
	uint64_t host_raw_ns;
	uint64_t sequence;
	uint64_t gpu_entry_ns;
};

enum launch_sample_status {
	LAUNCH_SAMPLE_CLASSIFIED = 0,
	LAUNCH_SAMPLE_UNCERTAIN = 1,
	LAUNCH_SAMPLE_CLOCK_ERROR = 2,
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

enum symbol_match_status {
	SYMBOL_ABSENT = 0,
	SYMBOL_UNDEFINED = 1,
	SYMBOL_NOT_FUNCTION = 2,
	SYMBOL_INVALID_VALUE = 3,
	SYMBOL_AMBIGUOUS = 4,
};

static int find_defined_symbol_matching(const char *path, const char *needle,
					enum symbol_match_status *status,
					uint64_t *vaddr);
static int find_x86_64_plt_entry(const char *path, const char *needle,
				 struct plt_entry *entry);

static int host_target_matches(uint64_t launch_ip, uint64_t func,
			       const struct host_target *target)
{
	uint64_t delta, expected;

	if (!target || !target->valid || !launch_ip)
		return -EINVAL;
	if (target->kernel_vaddr >= target->launch_vaddr) {
		delta = target->kernel_vaddr - target->launch_vaddr;
		if (delta > UINT64_MAX - launch_ip)
			return -ERANGE;
		expected = launch_ip + delta;
	} else {
		delta = target->launch_vaddr - target->kernel_vaddr;
		if (launch_ip < delta)
			return -ERANGE;
		expected = launch_ip - delta;
	}
	return func == expected;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t exiting;

static void sig_handler(int sig)
{
	exiting = true;
}

static int signed_difference(uint64_t left, uint64_t right, int64_t *value)
{
	if (left > INT64_MAX || right > INT64_MAX)
		return -ERANGE;
	*value = (int64_t)left - (int64_t)right;
	return 0;
}

static int subtract_i64(int64_t left, int64_t right, int64_t *value)
{
	if ((right > 0 && left < INT64_MIN + right) ||
	    (right < 0 && left > INT64_MAX + right))
		return -ERANGE;
	*value = left - right;
	return 0;
}

static uint64_t abs_i64(int64_t value)
{
	return value < 0 ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
}

static int calibration_valid(const struct clock_calibration *calibration)
{
	uint64_t width;

	if (!calibration || calibration->valid != 1 ||
	    !calibration->host_anchor_ns ||
	    calibration->offset_low_ns > calibration->offset_high_ns)
		return 0;
	width = (uint64_t)calibration->offset_high_ns -
		(uint64_t)calibration->offset_low_ns;
	return calibration->uncertainty_ns == width / 2 + width % 2;
}

static int consider_calibration_sample(struct clock_calibration *best,
				       uint64_t gpu_ns, uint64_t host_before_ns,
				       uint64_t host_after_ns)
{
	int64_t low, high;
	uint64_t width, best_width = UINT64_MAX;

	if (host_after_ns < host_before_ns)
		return -ERANGE;
	if (signed_difference(gpu_ns, host_after_ns, &low) ||
	    signed_difference(gpu_ns, host_before_ns, &high) || low > high)
		return -ERANGE;
	width = host_after_ns - host_before_ns;
	if (best->valid)
		best_width = (uint64_t)best->offset_high_ns -
			     (uint64_t)best->offset_low_ns;
	if (!best->valid || width < best_width) {
		best->offset_low_ns = low;
		best->offset_high_ns = high;
		best->uncertainty_ns = width / 2 + width % 2;
		best->host_anchor_ns = host_before_ns + width / 2;
		best->valid = 1;
	}
	return 0;
}

static int calibration_drift(const struct clock_calibration *start,
			     const struct clock_calibration *end,
			     struct clock_drift *drift)
{
	uint64_t largest_change;
	uint64_t scaled;

	memset(drift, 0, sizeof(*drift));
	if (!calibration_valid(start) || !calibration_valid(end) ||
	    end->host_anchor_ns <= start->host_anchor_ns ||
	    subtract_i64(end->offset_low_ns, start->offset_high_ns,
			 &drift->offset_change_low_ns) ||
	    subtract_i64(end->offset_high_ns, start->offset_low_ns,
			 &drift->offset_change_high_ns))
		return -EINVAL;
	drift->elapsed_ns = end->host_anchor_ns - start->host_anchor_ns;
	largest_change = abs_i64(drift->offset_change_low_ns);
	if (abs_i64(drift->offset_change_high_ns) > largest_change)
		largest_change = abs_i64(drift->offset_change_high_ns);
	if (largest_change > UINT64_MAX / 1000000000ULL)
		return -ERANGE;
	scaled = largest_change * 1000000000ULL;
	drift->rate_bound_ppb = scaled / drift->elapsed_ns +
		(scaled % drift->elapsed_ns != 0);
	drift->bounded = drift->rate_bound_ppb <= CLOCK_DRIFT_LIMIT_PPB;
	return 0;
}

static uint32_t histogram_bin(uint64_t delta_ns)
{
	if (delta_ns < 100)
		return 0;
	if (delta_ns < 1000)
		return 1;
	if (delta_ns < 10000)
		return 2;
	if (delta_ns < 100000)
		return 3;
	if (delta_ns < 1000000)
		return 4;
	if (delta_ns < 10000000)
		return 5;
	if (delta_ns < 100000000)
		return 6;
	if (delta_ns < 1000000000)
		return 7;
	if (delta_ns < 10000000000ULL)
		return 8;
	return 9;
}

static __int128 divide_floor(__int128 numerator, uint64_t denominator)
{
	__int128 quotient = numerator / denominator;
	if (numerator % denominator < 0)
		quotient--;
	return quotient;
}

static __int128 divide_ceil(__int128 numerator, uint64_t denominator)
{
	__int128 quotient = numerator / denominator;
	if (numerator % denominator > 0)
		quotient++;
	return quotient;
}

static int affine_offset_interval(uint64_t host_ns,
				  const struct clock_calibration *start,
				  const struct clock_calibration *end,
				  int64_t *low, int64_t *high)
{
	uint64_t elapsed, position;
	__int128 low_value, high_value;

	if (!calibration_valid(start) || !calibration_valid(end) ||
	    end->host_anchor_ns <= start->host_anchor_ns ||
	    host_ns < start->host_anchor_ns || host_ns > end->host_anchor_ns)
		return -ERANGE;
	elapsed = end->host_anchor_ns - start->host_anchor_ns;
	position = host_ns - start->host_anchor_ns;
	low_value = (__int128)start->offset_low_ns + divide_floor(
		((__int128)end->offset_low_ns - start->offset_low_ns) * position,
		elapsed);
	high_value = (__int128)start->offset_high_ns + divide_ceil(
		((__int128)end->offset_high_ns - start->offset_high_ns) * position,
		elapsed);
	if (low_value < INT64_MIN || low_value > INT64_MAX ||
	    high_value < INT64_MIN || high_value > INT64_MAX ||
	    low_value > high_value)
		return -ERANGE;
	*low = (int64_t)low_value;
	*high = (int64_t)high_value;
	return 0;
}

static enum launch_sample_status classify_affine_sample(
	const struct launch_sample *sample,
	const struct clock_calibration *start,
	const struct clock_calibration *end, uint32_t *bin)
{
	int64_t offset_low, offset_high, observed, latency_low, latency_high;

	if (!sample || !sample->host_raw_ns || !sample->gpu_entry_ns ||
	    signed_difference(sample->gpu_entry_ns, sample->host_raw_ns,
			      &observed) ||
	    affine_offset_interval(sample->host_raw_ns, start, end,
				   &offset_low, &offset_high) ||
	    subtract_i64(observed, offset_high, &latency_low) ||
	    subtract_i64(observed, offset_low, &latency_high) ||
	    latency_high < 0)
		return LAUNCH_SAMPLE_CLOCK_ERROR;
	if (latency_low < 0 ||
	    histogram_bin((uint64_t)latency_low) !=
		    histogram_bin((uint64_t)latency_high))
		return LAUNCH_SAMPLE_UNCERTAIN;
	*bin = histogram_bin((uint64_t)latency_low);
	return LAUNCH_SAMPLE_CLASSIFIED;
}

static int run_self_test(const char *elf_path, const char *kernel_symbol)
{
	struct clock_calibration calibration = {0};
	struct clock_calibration later = {0};
	struct clock_calibration affine_start = {
		.offset_low_ns = 100,
		.offset_high_ns = 120,
		.uncertainty_ns = 10,
		.host_anchor_ns = 1000000000ULL,
		.valid = 1,
	};
	struct clock_calibration affine_end = {
		.offset_low_ns = 300,
		.offset_high_ns = 340,
		.uncertainty_ns = 20,
		.host_anchor_ns = 1100000000ULL,
		.valid = 1,
	};
	struct clock_drift drift;
	struct launch_sample sample = {
		.host_raw_ns = 1050000000ULL,
		.sequence = 1,
		.gpu_entry_ns = 1050001250ULL,
	};
	struct host_target target = {0};
	struct plt_entry launch_plt = {0};
	enum symbol_match_status symbol_status;
	uint64_t target_vaddr, live_launch_ip, live_func;
	const uint64_t mock_load_bias = 0x100000000ULL;
	const char *test_path = elf_path ? elf_path : "/proc/self/exe";
	const char *test_kernel = kernel_symbol ? kernel_symbol : "run_self_test";
	const char *test_launch = elf_path ? CUDA_LAUNCH_SYMBOL : "printf";
	int64_t low, high;
	uint32_t bin;

	if (consider_calibration_sample(&calibration, 1250, 1000, 1100) ||
	    calibration.offset_low_ns != 150 ||
	    calibration.offset_high_ns != 250 ||
	    calibration.uncertainty_ns != 50 ||
	    calibration.host_anchor_ns != 1050)
		return 1;
	/* A narrower bracket must replace the first candidate. */
	if (consider_calibration_sample(&calibration, 2210, 2000, 2040) ||
	    calibration.offset_low_ns != 170 ||
	    calibration.offset_high_ns != 210 ||
	    calibration.uncertainty_ns != 20)
		return 1;
	if (consider_calibration_sample(&calibration, 1040, 800, 840) ||
	    calibration.offset_low_ns != 170 ||
	    calibration.offset_high_ns != 210)
		return 1;
	if (consider_calibration_sample(&later, 3200, 3000, 3040) ||
	    later.host_anchor_ns != 3020)
		return 1;
	/* Also exercise a negative GPU-minus-host offset. */
	memset(&calibration, 0, sizeof(calibration));
	if (consider_calibration_sample(&calibration, 900, 1000, 1040) ||
	    calibration.offset_low_ns != -140 ||
	    calibration.offset_high_ns != -100)
		return 1;

	if (calibration_drift(&affine_start, &affine_end, &drift) ||
	    drift.offset_change_low_ns != 180 ||
	    drift.offset_change_high_ns != 240 ||
	    drift.elapsed_ns != 100000000ULL ||
	    drift.rate_bound_ppb != 2400 || !drift.bounded ||
	    affine_offset_interval(sample.host_raw_ns, &affine_start,
				   &affine_end, &low, &high) ||
	    low != 200 || high != 230 ||
	    classify_affine_sample(&sample, &affine_start, &affine_end, &bin) !=
		    LAUNCH_SAMPLE_CLASSIFIED ||
	    bin != 2)
		return 1;
	/* A full interval crossing 1 us is retained as uncertainty. */
	sample.gpu_entry_ns = sample.host_raw_ns + 1210;
	if (classify_affine_sample(&sample, &affine_start, &affine_end, &bin) !=
	    LAUNCH_SAMPLE_UNCERTAIN)
		return 1;
	/* Causally impossible affine intervals remain true clock errors. */
	sample.gpu_entry_ns = sample.host_raw_ns + 100;
	if (classify_affine_sample(&sample, &affine_start, &affine_end, &bin) !=
	    LAUNCH_SAMPLE_CLOCK_ERROR)
		return 1;
	affine_end.offset_low_ns = 2000100;
	affine_end.offset_high_ns = 2000140;
	if (calibration_drift(&affine_start, &affine_end, &drift) ||
	    drift.bounded || drift.rate_bound_ppb <= CLOCK_DRIFT_LIMIT_PPB)
		return 1;

	/* Exercise the same-ELF PLT-to-local-symbol address proof without CUDA. */
	if (find_defined_symbol_matching(test_path, test_kernel, &symbol_status,
					 &target_vaddr) ||
	    find_x86_64_plt_entry(test_path, test_launch, &launch_plt) ||
	    !launch_plt.file_offset || !launch_plt.vaddr || !target_vaddr ||
	    launch_plt.vaddr > UINT64_MAX - mock_load_bias ||
	    target_vaddr > UINT64_MAX - mock_load_bias)
		return 1;
	target.launch_vaddr = launch_plt.vaddr;
	target.kernel_vaddr = target_vaddr;
	target.valid = 1;
	live_launch_ip = mock_load_bias + launch_plt.vaddr;
	live_func = mock_load_bias + target_vaddr;
	if (host_target_matches(live_launch_ip, live_func, &target) != 1 ||
	    host_target_matches(live_launch_ip, live_func + 1, &target) != 0)
		return 1;
	target.valid = 0;
	if (host_target_matches(live_launch_ip, live_func, &target) >= 0)
		return 1;
	target.valid = 1;
	target.kernel_vaddr = UINT64_MAX;
	if (host_target_matches(live_launch_ip, live_func, &target) != -ERANGE)
		return 1;

	if (rm_ptimer_575_self_test())
		return 1;
	printf("launchlate CPU self-test: PASS\n");
	return 0;
}

static int calibrate_gpu_clock(struct clock_calibration *calibration,
			       struct rm_calibration_run *run)
{
	struct rm_ptimer_575_client client;
	struct rm_ptimer_575_sample sample;
	int err = 0, cleanup_err;
	unsigned int trial;

	memset(calibration, 0, sizeof(*calibration));
	memset(run, 0, sizeof(*run));
	run->requested = CALIBRATION_TRIALS;
	rm_ptimer_575_client_init(&client);
	err = rm_ptimer_575_open(&client);
	if (err)
		return err;
	for (trial = 0; trial < CALIBRATION_WARMUPS; trial++) {
		err = rm_ptimer_575_sample(&client, &sample);
		if (err)
			goto cleanup;
	}
	for (trial = 0; trial < CALIBRATION_TRIALS; trial++) {
		err = rm_ptimer_575_sample(&client, &sample);
		if (err) {
			run->rejected++;
			goto cleanup;
		}
		run->accepted++;
		if (!calibration->valid ||
		    sample.bracket_width_ns < run->best.bracket_width_ns) {
			run->best = sample;
			calibration->offset_low_ns = sample.offset_low_ns;
			calibration->offset_high_ns = sample.offset_high_ns;
			calibration->uncertainty_ns =
				sample.bracket_width_ns / 2 +
				sample.bracket_width_ns % 2;
			calibration->host_anchor_ns = sample.cpu_before_raw_ns +
				sample.selected_gap_ns / 2;
			calibration->valid = 1;
		}
	}
	if (!calibration_valid(calibration))
		err = -ERANGE;

cleanup:
	cleanup_err = rm_ptimer_575_close(&client);
	run->cleanup_complete = cleanup_err == 0;
	if (cleanup_err && !err)
		err = cleanup_err;
	return err;
}

static void print_calibration(const char *phase,
			      const struct clock_calibration *calibration,
			      const struct rm_calibration_run *run)
{
	printf("%s clock offset lower: %" PRId64 " ns\n", phase,
	       calibration->offset_low_ns);
	printf("%s clock offset upper: %" PRId64 " ns\n", phase,
	       calibration->offset_high_ns);
	printf("%s clock uncertainty: %" PRIu64 " ns\n", phase,
	       calibration->uncertainty_ns);
	printf("%s clock host anchor: %" PRIu64 " ns\n", phase,
	       calibration->host_anchor_ns);
	printf("%s RM samples requested: %" PRIu64 "\n", phase,
	       run->requested);
	printf("%s RM samples accepted: %" PRIu64 "\n", phase, run->accepted);
	printf("%s RM samples rejected: %" PRIu64 "\n", phase, run->rejected);
	printf("%s RM outer before RAW: %" PRIu64 " ns\n", phase,
	       run->best.outer_before_raw_ns);
	printf("%s RM CPU before RAW: %" PRIu64 " ns\n", phase,
	       run->best.cpu_before_raw_ns);
	printf("%s RM GPU PTIMER: %" PRIu64 " ns\n", phase,
	       run->best.gpu_ptimer_ns);
	printf("%s RM CPU after RAW: %" PRIu64 " ns\n", phase,
	       run->best.cpu_after_raw_ns);
	printf("%s RM outer after RAW: %" PRIu64 " ns\n", phase,
	       run->best.outer_after_raw_ns);
	printf("%s RM outer width: %" PRIu64 " ns\n", phase,
	       run->best.outer_width_ns);
	printf("%s RM selected gap: %" PRIu64 " ns\n", phase,
	       run->best.selected_gap_ns);
	printf("%s RM bracket width: %" PRIu64 " ns\n", phase,
	       run->best.bracket_width_ns);
	printf("%s RM status: 0x%08" PRIx32 "\n", phase,
	       run->best.rm_status);
	printf("%s RM cleanup complete: %" PRIu64 "\n", phase,
	       run->cleanup_complete);
}

static int detach_probes(struct launchlate_bpf *skel)
{
	int err = 0;
	int result;

	if (skel->links.uprobe_cuda_launch) {
		result = bpf_link__destroy(skel->links.uprobe_cuda_launch);
		skel->links.uprobe_cuda_launch = NULL;
		if (result && !err)
			err = result;
	}
	if (skel->links.cuda__probe) {
		result = bpf_link__destroy(skel->links.cuda__probe);
		skel->links.cuda__probe = NULL;
		if (result && !err)
			err = result;
	}
	return err;
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

static int print_histogram(struct launchlate_bpf *obj,
			   const struct clock_calibration *start,
			   const struct clock_calibration *end)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	uint32_t i;
	uint64_t histogram[HIST_BINS] = {0};
	uint64_t online_histogram_total = 0;
	uint64_t classified = 0, uncertain = 0, clock_errors = 0;
	uint64_t histogram_total = 0, value;
	int histogram_fd = bpf_map__fd(obj->maps.time_histogram);
	int launch_fd = bpf_map__fd(obj->maps.launch_times);
	int host_fd = bpf_map__fd(obj->maps.host_state);
	int device_fd = bpf_map__fd(obj->maps.device_state);
	uint64_t host_values[HOST_COUNTERS] = {0};
	uint64_t device_values[DEVICE_COUNTERS] = {0};
	int err = 0;

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

	/* Validate the online device-side path before replacing its stale model. */
	for (i = 0; i < HIST_BINS; i++) {
		err = bpf_map_lookup_elem(histogram_fd, &i, &value);
		if (err && errno != ENOENT) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		if (!err)
			online_histogram_total += value;
	}
	for (i = 0; i < HOST_COUNTERS; i++) {
		uint32_t state_key = i;
		if (bpf_map_lookup_elem(host_fd, &state_key, &host_values[i]) != 0) {
			warn("host_state lookup failed: %s\n", strerror(errno));
			return -1;
		}
	}
	for (i = 0; i < DEVICE_COUNTERS; i++) {
		uint32_t state_key = i;
		if (bpf_map_lookup_elem(device_fd, &state_key,
					&device_values[i]) != 0) {
			warn("device_state lookup failed: %s\n", strerror(errno));
			return -1;
		}
	}

	/*
	 * The end anchor is unavailable inside the live GPU callback.  Reclassify
	 * the retained exact pairs against the affine interpolation of both
	 * bracketed offset intervals; never use an offset midpoint as exact.
	 */
	if (device_values[MATCHED_SAMPLES] > LAUNCH_QUEUE_SIZE) {
		clock_errors = device_values[MATCHED_SAMPLES];
	} else {
		for (uint64_t sequence = 0;
		     sequence < device_values[MATCHED_SAMPLES]; sequence++) {
			uint32_t slot = (uint32_t)sequence;
			uint32_t bin = 0;
			struct launch_sample sample = {0};
			enum launch_sample_status status;

			if (bpf_map_lookup_elem(launch_fd, &slot, &sample) != 0 ||
			    sample.sequence != sequence + 1) {
				clock_errors++;
				continue;
			}
			status = classify_affine_sample(&sample, start, end, &bin);
			if (status == LAUNCH_SAMPLE_CLASSIFIED) {
				histogram[bin]++;
				classified++;
			} else if (status == LAUNCH_SAMPLE_UNCERTAIN) {
				uncertain++;
			} else {
				clock_errors++;
			}
		}
	}
	for (i = 0; i < HIST_BINS; i++)
		histogram_total += histogram[i];

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);
	printf("\n%-9s Launch Latency Distribution:\n", ts);
	printf("%-15s : count    distribution\n", "latency");
	for (i = 0; i < HIST_BINS; i++) {
		if (!histogram[i])
			continue;
		printf("%-15s : %-8" PRIu64 " |", labels[i], histogram[i]);
		int bar_len = (histogram[i] * 40) /
			      (histogram_total ? histogram_total : 1);
		if (!bar_len)
			bar_len = 1;
		for (int j = 0; j < bar_len; j++)
			printf("*");
		printf("\n");
	}

	printf("Histogram samples: %" PRIu64 "\n", histogram_total);
	printf("Total samples: %" PRIu64 "\n", device_values[MATCHED_SAMPLES]);
	printf("Host launches: %" PRIu64 "\n", host_values[HOST_LAUNCHES]);
	printf("Host enqueued: %" PRIu64 "\n", host_values[HOST_ENQUEUED]);
	printf("Host launch calls: %" PRIu64 "\n",
	       host_values[HOST_LAUNCH_CALLS]);
	printf("Host target errors: %" PRIu64 "\n",
	       host_values[HOST_TARGET_ERRORS]);
	printf("Device entries: %" PRIu64 "\n", device_values[DEVICE_ENTRIES]);
	printf("Matched samples: %" PRIu64 "\n", device_values[MATCHED_SAMPLES]);
	printf("Queue underflows: %" PRIu64 "\n",
	       device_values[QUEUE_UNDERFLOWS]);
	printf("Queue overflows: %" PRIu64 "\n", host_values[QUEUE_OVERFLOWS]);
	printf("Queue update errors: %" PRIu64 "\n",
	       host_values[QUEUE_UPDATE_ERRORS]);
	printf("Online classified samples: %" PRIu64 "\n",
	       device_values[CLASSIFIED_SAMPLES]);
	printf("Online uncertain samples: %" PRIu64 "\n",
	       device_values[UNCERTAIN_SAMPLES]);
	printf("Online clock errors: %" PRIu64 "\n", device_values[CLOCK_ERRORS]);
	printf("Classified samples: %" PRIu64 "\n", classified);
	printf("Uncertain samples: %" PRIu64 "\n", uncertain);
	printf("Clock errors: %" PRIu64 "\n", clock_errors);
	bool online_accounting_complete =
		device_values[MATCHED_SAMPLES] ==
			device_values[CLASSIFIED_SAMPLES] +
			device_values[UNCERTAIN_SAMPLES] +
			device_values[CLOCK_ERRORS] &&
		device_values[CLASSIFIED_SAMPLES] == online_histogram_total;
	bool accounting_complete =
		host_values[HOST_LAUNCH_CALLS] >= host_values[HOST_LAUNCHES] &&
		host_values[HOST_LAUNCHES] == host_values[HOST_ENQUEUED] +
			host_values[QUEUE_OVERFLOWS] +
			host_values[QUEUE_UPDATE_ERRORS] &&
		device_values[DEVICE_ENTRIES] ==
			device_values[MATCHED_SAMPLES] +
			device_values[QUEUE_UNDERFLOWS] &&
		device_values[MATCHED_SAMPLES] ==
			classified + uncertain + clock_errors &&
		classified == histogram_total;
	bool pairing_complete =
		host_values[HOST_LAUNCHES] > 0 &&
		host_values[HOST_LAUNCHES] == host_values[HOST_ENQUEUED] &&
		host_values[HOST_ENQUEUED] == device_values[DEVICE_ENTRIES] &&
		device_values[DEVICE_ENTRIES] == device_values[MATCHED_SAMPLES] &&
		host_values[HOST_TARGET_ERRORS] == 0 &&
		host_values[QUEUE_OVERFLOWS] == 0 &&
		host_values[QUEUE_UPDATE_ERRORS] == 0 &&
		device_values[QUEUE_UNDERFLOWS] == 0 &&
		device_values[CLOCK_ERRORS] == 0 && clock_errors == 0;
	printf("Online accounting complete: %d\n",
	       online_accounting_complete ? 1 : 0);
	printf("Accounting complete: %d\n", accounting_complete ? 1 : 0);
	printf("Pairing complete: %d\n", pairing_complete ? 1 : 0);
	fflush(stdout);
	return online_accounting_complete && accounting_complete && pairing_complete ?
		0 : -EIO;
}

int main(int argc, char **argv)
{
	struct launchlate_bpf *skel;
	struct clock_calibration start_calibration = {0};
	struct clock_calibration end_calibration = {0};
	struct rm_calibration_run start_rm = {0};
	struct rm_calibration_run end_rm = {0};
	struct clock_drift drift = {0};
	struct host_target target = {0};
	struct plt_entry launch_plt = {0};
	int err, stat_err;
	uint32_t key = 0;
	const char *binary_path = "./vec_add";
	const char *symbol_hint = DEFAULT_UPROBE_SYMBOL_HINT;
	enum symbol_match_status symbol_status;

	if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
		if (argc != 2 && argc != 4) {
			fprintf(stderr,
				"Usage: %s --self-test [target-elf target-symbol]\n",
				argv[0]);
			return 1;
		}
		return run_self_test(argc == 4 ? argv[2] : NULL,
				     argc == 4 ? argv[3] : NULL);
	}
	if (argc > 1)
		binary_path = argv[1];
	if (argc > 2)
		symbol_hint = argv[2];

	err = find_defined_symbol_matching(binary_path, symbol_hint, &symbol_status,
					   &target.kernel_vaddr);
	if (err) {
		const char *reason = symbol_status == SYMBOL_UNDEFINED ?
			"the exact symbol exists only as an undefined import" :
			symbol_status == SYMBOL_NOT_FUNCTION ?
			"the exact defined symbol is not an ELF function" :
			symbol_status == SYMBOL_INVALID_VALUE ?
			"the exact defined function has no usable virtual address" :
			symbol_status == SYMBOL_AMBIGUOUS ?
			"the exact name resolves to multiple function addresses" :
			"the exact symbol is absent";
		fprintf(stderr, "Cannot attach '%s' in %s: %s\n", symbol_hint,
			binary_path, reason);
		return 1;
	}
	err = find_x86_64_plt_entry(binary_path, CUDA_LAUNCH_SYMBOL,
				    &launch_plt);
	if (err || launch_plt.file_offset > SIZE_MAX) {
		const char *reason = err == -ENOTSUP ?
			"only x86-64 ELF PLT layouts are supported" :
			err == -EEXIST ?
			"multiple matching PLT relocations exist" :
			"no unique valid PLT relocation exists";
		fprintf(stderr, "Cannot attach '%s@plt' in %s: %s\n",
			CUDA_LAUNCH_SYMBOL, binary_path, reason);
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
	err = bpf_map_update_elem(bpf_map__fd(skel->maps.host_target), &key,
				  &target, BPF_ANY);
	if (err) {
		err = -errno;
		fprintf(stderr, "Failed to update host_target map: %s\n",
			strerror(errno));
		goto cleanup;
	}

	/*
	 * The versioned RM zipper returns the selected CLOCK_MONOTONIC_RAW
	 * endpoints around a PTIMER/global-timer read.  The host callback uses the
	 * same RAW clock; final classification converts it to a PTIMER interval.
	 */
	err = calibrate_gpu_clock(&start_calibration, &start_rm);
	print_calibration("Start", &start_calibration, &start_rm);
	if (err) {
		fprintf(stderr, "Failed to calibrate RM PTIMER and RAW clocks: %s\n",
			strerror(-err));
		goto cleanup;
	}
	printf("Clock calibration method: RM endpoints-v1 PTIMER intervals with affine CLOCK_MONOTONIC_RAW interpolation\n");
	err = bpf_map_update_elem(
		bpf_map__fd(skel->maps.clock_calibration), &key,
		&start_calibration, BPF_ANY);
	if (err) {
		err = -errno;
		fprintf(stderr, "Failed to update clock_calibration map: %s\n",
			strerror(errno));
		goto cleanup;
	}

	printf("Attaching uprobe: binary_path='%s', target='%s', launch='%s@plt' (exact same-ELF address match)\n",
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

	stat_err = detach_probes(skel);
	printf("Probes detached before final readback: %d\n",
	       stat_err == 0 ? 1 : 0);
	if (stat_err) {
		fprintf(stderr, "Failed to detach launch probes: %s\n",
			strerror(-stat_err));
		err = stat_err;
	}

	stat_err = calibrate_gpu_clock(&end_calibration, &end_rm);
	print_calibration("End", &end_calibration, &end_rm);
	if (stat_err) {
		fprintf(stderr, "Failed to validate RM PTIMER and RAW clocks: %s\n",
			strerror(-stat_err));
		if (!err)
			err = stat_err;
	} else {
		stat_err = calibration_drift(&start_calibration, &end_calibration,
					     &drift);
		printf("Clock offset change lower: %" PRId64 " ns\n",
		       drift.offset_change_low_ns);
		printf("Clock offset change upper: %" PRId64 " ns\n",
		       drift.offset_change_high_ns);
		printf("Clock calibration elapsed: %" PRIu64 " ns\n",
		       drift.elapsed_ns);
		printf("Clock drift rate bound: %" PRIu64 " ppb\n",
		       drift.rate_bound_ppb);
		printf("Clock drift limit: %" PRIu64 " ppb\n",
		       (uint64_t)CLOCK_DRIFT_LIMIT_PPB);
		printf("Clock drift bounded: %d\n",
		       stat_err == 0 && drift.bounded ? 1 : 0);
		if (stat_err || !drift.bounded) {
			fprintf(stderr, "Clock calibration drift exceeds its bound\n");
			if (!err)
				err = stat_err ? stat_err : -ERANGE;
		}
	}
	stat_err = print_histogram(skel, &start_calibration, &end_calibration);
	if (!err && stat_err)
		err = stat_err;

cleanup:
	/* Clean up */
	launchlate_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
