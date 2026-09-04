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
#include <dlfcn.h>
#include <limits.h>
#include <gelf.h>
#include "./.output/launchlate.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

#define DEFAULT_UPROBE_SYMBOL_HINT "_Z9vectorAddPKfS0_Pf"
#define HIST_BINS 10
#define HOST_COUNTERS 4
#define DEVICE_COUNTERS 6
#define CALIBRATION_WARMUPS 4
#define CALIBRATION_TRIALS 32

enum host_counter {
	HOST_LAUNCHES = 0,
	HOST_ENQUEUED = 1,
	QUEUE_OVERFLOWS = 2,
	QUEUE_UPDATE_ERRORS = 3,
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
	uint64_t valid;
};

typedef int ll_cu_result;
typedef struct CUctx_st *ll_cu_context;
typedef struct CUmod_st *ll_cu_module;
typedef struct CUfunc_st *ll_cu_function;
typedef unsigned long long ll_cu_device_ptr;
typedef struct CUstream_st *ll_cu_stream;

struct cuda_driver_api {
	void *library;
	ll_cu_result (*init)(unsigned int);
	ll_cu_result (*ctx_get_current)(ll_cu_context *);
	ll_cu_result (*module_load_data)(ll_cu_module *, const void *);
	ll_cu_result (*module_get_function)(ll_cu_function *, ll_cu_module,
					   const char *);
	ll_cu_result (*mem_alloc)(ll_cu_device_ptr *, size_t);
	ll_cu_result (*mem_free)(ll_cu_device_ptr);
	ll_cu_result (*launch_kernel)(ll_cu_function, unsigned int, unsigned int,
				      unsigned int, unsigned int, unsigned int,
				      unsigned int, unsigned int, ll_cu_stream,
				      void **, void **);
	ll_cu_result (*ctx_synchronize)(void);
	ll_cu_result (*memcpy_dtoh)(void *, ll_cu_device_ptr, size_t);
	ll_cu_result (*module_unload)(ll_cu_module);
};

static const char calibration_ptx[] =
	".version 7.0\n"
	".target sm_60\n"
	".address_size 64\n"
	".visible .entry launchlate_clock_sample(\n"
	"    .param .u64 output_ptr\n"
	")\n"
	"{\n"
	"    .reg .b64 %rd<3>;\n"
	"    ld.param.u64 %rd1, [output_ptr];\n"
	"    mov.u64 %rd2, %globaltimer;\n"
	"    st.global.u64 [%rd1], %rd2;\n"
	"    ret;\n"
	"}\n";

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

static int monotonic_ns(uint64_t *value)
{
	struct timespec now;

	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
		return -errno;
	if (now.tv_sec < 0 || (uint64_t)now.tv_sec > UINT64_MAX / 1000000000ULL)
		return -ERANGE;
	*value = (uint64_t)now.tv_sec * 1000000000ULL +
		 (uint64_t)now.tv_nsec;
	return 0;
}

static int signed_difference(uint64_t left, uint64_t right, int64_t *value)
{
	if (left > INT64_MAX || right > INT64_MAX)
		return -ERANGE;
	*value = (int64_t)left - (int64_t)right;
	return 0;
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
		best->valid = 1;
	}
	return 0;
}

static int calibration_intersection(const struct clock_calibration *first,
				    const struct clock_calibration *second,
				    int64_t *low, int64_t *high)
{
	if (!first->valid || !second->valid)
		return -EINVAL;
	*low = first->offset_low_ns > second->offset_low_ns ?
		first->offset_low_ns : second->offset_low_ns;
	*high = first->offset_high_ns < second->offset_high_ns ?
		first->offset_high_ns : second->offset_high_ns;
	return *low <= *high ? 0 : -ERANGE;
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

static int latency_interval(uint64_t host_ns, uint64_t gpu_ns,
			    const struct clock_calibration *calibration,
			    int64_t *low, int64_t *high)
{
	int64_t observed;

	if (!calibration->valid ||
	    calibration->offset_low_ns > calibration->offset_high_ns ||
	    signed_difference(gpu_ns, host_ns, &observed))
		return -ERANGE;
	*low = observed - calibration->offset_high_ns;
	*high = observed - calibration->offset_low_ns;
	return *high < 0 ? -ERANGE : 0;
}

static int run_self_test(void)
{
	struct clock_calibration calibration = {0};
	struct clock_calibration later = {0};
	int64_t low, high;

	if (consider_calibration_sample(&calibration, 1250, 1000, 1100) ||
	    calibration.offset_low_ns != 150 ||
	    calibration.offset_high_ns != 250 ||
	    calibration.uncertainty_ns != 50)
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
	if (latency_interval(5000, 5400, &calibration, &low, &high) ||
	    low != 190 || high != 230 ||
	    histogram_bin((uint64_t)low) != histogram_bin((uint64_t)high))
		return 1;
	if (consider_calibration_sample(&later, 3200, 3000, 3040) ||
	    calibration_intersection(&calibration, &later, &low, &high) ||
	    low != 170 || high != 200)
		return 1;
	/* Also exercise a negative GPU-minus-host offset. */
	memset(&calibration, 0, sizeof(calibration));
	if (consider_calibration_sample(&calibration, 900, 1000, 1040) ||
	    latency_interval(2000, 1975, &calibration, &low, &high) ||
	    low != 75 || high != 115)
		return 1;

	printf("launchlate CPU self-test: PASS\n");
	return 0;
}

static int load_cuda_driver(struct cuda_driver_api *api)
{
	void *symbol;
	const char *error;

	memset(api, 0, sizeof(*api));
	api->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!api->library) {
		warn("Clock calibration could not load libcuda.so.1: %s\n",
		     dlerror());
		return -ENOENT;
	}

#define LOAD_CUDA(member, name)                                                   \
	do {                                                                         \
		dlerror();                                                            \
		symbol = dlsym(api->library, name);                                    \
		error = dlerror();                                                     \
		if (error || !symbol) {                                                \
			warn("Clock calibration missing CUDA symbol %s: %s\n", name,   \
			     error ? error : "unknown error");                         \
			dlclose(api->library);                                          \
			memset(api, 0, sizeof(*api));                                   \
			return -ENOSYS;                                                 \
		}                                                                    \
		memcpy(&api->member, &symbol, sizeof(symbol));                         \
	} while (0)

	LOAD_CUDA(init, "cuInit");
	LOAD_CUDA(ctx_get_current, "cuCtxGetCurrent");
	LOAD_CUDA(module_load_data, "cuModuleLoadData");
	LOAD_CUDA(module_get_function, "cuModuleGetFunction");
	LOAD_CUDA(mem_alloc, "cuMemAlloc_v2");
	LOAD_CUDA(mem_free, "cuMemFree_v2");
	LOAD_CUDA(launch_kernel, "cuLaunchKernel");
	LOAD_CUDA(ctx_synchronize, "cuCtxSynchronize");
	LOAD_CUDA(memcpy_dtoh, "cuMemcpyDtoH_v2");
	LOAD_CUDA(module_unload, "cuModuleUnload");
#undef LOAD_CUDA
	return 0;
}

static int cuda_call(ll_cu_result result, const char *operation)
{
	if (result == 0)
		return 0;
	warn("Clock calibration CUDA operation %s failed: %d\n", operation,
	     result);
	return -EIO;
}

static int sample_gpu_clock(struct cuda_driver_api *api,
			    ll_cu_function function, ll_cu_device_ptr output,
			    uint64_t *gpu_ns, uint64_t *host_before_ns,
			    uint64_t *host_after_ns)
{
	void *arguments[] = {&output};
	int err;

	err = monotonic_ns(host_before_ns);
	if (err)
		return err;
	err = cuda_call(api->launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
					   NULL, arguments, NULL),
			"cuLaunchKernel");
	if (err)
		return err;
	err = cuda_call(api->ctx_synchronize(), "cuCtxSynchronize");
	if (err)
		return err;
	err = monotonic_ns(host_after_ns);
	if (err)
		return err;
	err = cuda_call(api->memcpy_dtoh(gpu_ns, output, sizeof(*gpu_ns)),
			"cuMemcpyDtoH");
	if (err)
		return err;
	return *gpu_ns ? 0 : -ERANGE;
}

static int calibrate_gpu_clock(struct clock_calibration *calibration)
{
	struct cuda_driver_api api;
	ll_cu_context context = NULL;
	ll_cu_module module = NULL;
	ll_cu_function function = NULL;
	ll_cu_device_ptr output = 0;
	uint64_t gpu_ns, before_ns, after_ns;
	int err, cleanup_err = 0;
	unsigned int trial;

	memset(calibration, 0, sizeof(*calibration));
	err = load_cuda_driver(&api);
	if (err)
		return err;
	if ((err = cuda_call(api.init(0), "cuInit")) ||
	    (err = cuda_call(api.ctx_get_current(&context), "cuCtxGetCurrent")))
		goto cleanup;
	/* GPU-backed BPF maps create the owner context during skeleton load. */
	if (!context) {
		warn("Clock calibration requires the BPF map owner CUDA context\n");
		err = -ENODEV;
		goto cleanup;
	}
	if ((err = cuda_call(api.module_load_data(&module, calibration_ptx),
			     "cuModuleLoadData")) ||
	    (err = cuda_call(api.module_get_function(
				&function, module, "launchlate_clock_sample"),
			     "cuModuleGetFunction")) ||
	    (err = cuda_call(api.mem_alloc(&output, sizeof(gpu_ns)),
			     "cuMemAlloc")))
		goto cleanup;

	for (trial = 0; trial < CALIBRATION_WARMUPS; trial++) {
		err = sample_gpu_clock(&api, function, output, &gpu_ns,
				       &before_ns, &after_ns);
		if (err)
			goto cleanup;
	}
	for (trial = 0; trial < CALIBRATION_TRIALS; trial++) {
		err = sample_gpu_clock(&api, function, output, &gpu_ns,
				       &before_ns, &after_ns);
		if (err || (err = consider_calibration_sample(
				calibration, gpu_ns, before_ns, after_ns)))
			goto cleanup;
	}
	if (!calibration->valid)
		err = -ERANGE;

cleanup:
	if (output && (cleanup_err = cuda_call(api.mem_free(output), "cuMemFree")) &&
	    !err)
		err = cleanup_err;
	if (module &&
	    (cleanup_err = cuda_call(api.module_unload(module), "cuModuleUnload")) &&
	    !err)
		err = cleanup_err;
	dlclose(api.library);
	return err;
}

static void print_calibration(const char *phase,
			      const struct clock_calibration *calibration)
{
	printf("%s clock offset lower: %" PRId64 " ns\n", phase,
	       calibration->offset_low_ns);
	printf("%s clock offset upper: %" PRId64 " ns\n", phase,
	       calibration->offset_high_ns);
	printf("%s clock uncertainty: %" PRIu64 " ns\n", phase,
	       calibration->uncertainty_ns);
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

enum symbol_match_status {
	SYMBOL_ABSENT = 0,
	SYMBOL_UNDEFINED = 1,
	SYMBOL_NOT_FUNCTION = 2,
};

static char *find_defined_symbol_matching(const char *path, const char *needle,
					  enum symbol_match_status *status)
{
	Elf *e = NULL;
	Elf_Scn *scn = NULL;
	Elf_Data *data = NULL;
	GElf_Shdr shdr;
	GElf_Sym sym;
	int fd = -1;

	*status = SYMBOL_ABSENT;
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
	uint64_t histogram_total = 0;

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
	for (i = 0; i < HIST_BINS; i++) {
		err = bpf_map_lookup_elem(fd, &i, &value);
		if (err && errno != ENOENT) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		if (!err && value > 0) {
			histogram_total += value;
		}
	}

	// Print histogram
	for (i = 0; i < HIST_BINS; i++) {
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
			int bar_len = (value * 40) /
				      (histogram_total > 0 ? histogram_total : 1);
			if (bar_len == 0 && value > 0)
				bar_len = 1;
			for (int j = 0; j < bar_len; j++)
				printf("*");
			printf("\n");
		}
	}

	printf("Histogram samples: %" PRIu64 "\n", histogram_total);
	int host_fd = bpf_map__fd(obj->maps.host_state);
	int device_fd = bpf_map__fd(obj->maps.device_state);
	uint64_t host_values[HOST_COUNTERS] = {0};
	uint64_t device_values[DEVICE_COUNTERS] = {0};
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
	printf("Total samples: %" PRIu64 "\n", device_values[MATCHED_SAMPLES]);
	printf("Host launches: %" PRIu64 "\n", host_values[HOST_LAUNCHES]);
	printf("Host enqueued: %" PRIu64 "\n", host_values[HOST_ENQUEUED]);
	printf("Device entries: %" PRIu64 "\n", device_values[DEVICE_ENTRIES]);
	printf("Matched samples: %" PRIu64 "\n", device_values[MATCHED_SAMPLES]);
	printf("Queue underflows: %" PRIu64 "\n",
	       device_values[QUEUE_UNDERFLOWS]);
	printf("Queue overflows: %" PRIu64 "\n", host_values[QUEUE_OVERFLOWS]);
	printf("Queue update errors: %" PRIu64 "\n",
	       host_values[QUEUE_UPDATE_ERRORS]);
	printf("Classified samples: %" PRIu64 "\n",
	       device_values[CLASSIFIED_SAMPLES]);
	printf("Uncertain samples: %" PRIu64 "\n",
	       device_values[UNCERTAIN_SAMPLES]);
	printf("Clock errors: %" PRIu64 "\n", device_values[CLOCK_ERRORS]);
	bool accounting_complete =
		host_values[HOST_LAUNCHES] == host_values[HOST_ENQUEUED] +
			host_values[QUEUE_OVERFLOWS] +
			host_values[QUEUE_UPDATE_ERRORS] &&
		device_values[DEVICE_ENTRIES] ==
			device_values[MATCHED_SAMPLES] +
			device_values[QUEUE_UNDERFLOWS] &&
		device_values[MATCHED_SAMPLES] ==
			device_values[CLASSIFIED_SAMPLES] +
			device_values[UNCERTAIN_SAMPLES] +
			device_values[CLOCK_ERRORS] &&
		device_values[CLASSIFIED_SAMPLES] == histogram_total;
	bool pairing_complete =
		host_values[HOST_LAUNCHES] > 0 &&
		host_values[HOST_LAUNCHES] == host_values[HOST_ENQUEUED] &&
		host_values[HOST_ENQUEUED] == device_values[DEVICE_ENTRIES] &&
		device_values[DEVICE_ENTRIES] == device_values[MATCHED_SAMPLES] &&
		host_values[QUEUE_OVERFLOWS] == 0 &&
		host_values[QUEUE_UPDATE_ERRORS] == 0 &&
		device_values[QUEUE_UNDERFLOWS] == 0 &&
		device_values[CLOCK_ERRORS] == 0;
	printf("Accounting complete: %d\n", accounting_complete ? 1 : 0);
	printf("Pairing complete: %d\n", pairing_complete ? 1 : 0);
	fflush(stdout);
	return accounting_complete && pairing_complete ? 0 : -EIO;
}

int main(int argc, char **argv)
{
	struct launchlate_bpf *skel;
	struct clock_calibration start_calibration = {0};
	struct clock_calibration end_calibration = {0};
	int64_t intersection_low = 0, intersection_high = 0;
	int err, stat_err;
	uint32_t key = 0;
	const char *binary_path = "./vec_add";
	const char *symbol_hint = DEFAULT_UPROBE_SYMBOL_HINT;
	enum symbol_match_status symbol_status;
	char *func_name = NULL;

	if (argc == 2 && strcmp(argv[1], "--self-test") == 0)
		return run_self_test();
	if (argc > 1)
		binary_path = argv[1];
	if (argc > 2)
		symbol_hint = argv[2];

	func_name = find_defined_symbol_matching(binary_path, symbol_hint,
					 &symbol_status);
	if (!func_name) {
		const char *reason = symbol_status == SYMBOL_UNDEFINED ?
			"the exact symbol exists only as an undefined import" :
			symbol_status == SYMBOL_NOT_FUNCTION ?
			"the exact defined symbol is not an ELF function" :
			"the exact symbol is absent";
		fprintf(stderr, "Cannot attach '%s' in %s: %s\n", symbol_hint,
			binary_path, reason);
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

	/*
	 * Bracket a real %globaltimer read between two CLOCK_MONOTONIC reads.
	 * This yields an offset interval and an explicit uncertainty bound; no
	 * realtime/globaltimer epoch assumption or negative-delta clamp is used.
	 */
	err = calibrate_gpu_clock(&start_calibration);
	if (err) {
		fprintf(stderr, "Failed to calibrate GPU and host clocks: %s\n",
			strerror(-err));
		goto cleanup;
	}
	printf("Clock calibration method: bracketed %%globaltimer kernel against CLOCK_MONOTONIC\n");
	print_calibration("Start", &start_calibration);
	err = bpf_map_update_elem(
		bpf_map__fd(skel->maps.clock_calibration), &key,
		&start_calibration, BPF_ANY);
	if (err) {
		err = -errno;
		fprintf(stderr, "Failed to update clock_calibration map: %s\n",
			strerror(errno));
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

	stat_err = calibrate_gpu_clock(&end_calibration);
	if (stat_err) {
		fprintf(stderr, "Failed to validate GPU and host clocks: %s\n",
			strerror(-stat_err));
		if (!err)
			err = stat_err;
	} else {
		print_calibration("End", &end_calibration);
		stat_err = calibration_intersection(
			&start_calibration, &end_calibration,
			&intersection_low, &intersection_high);
		printf("Clock calibration endpoint overlap: %d\n",
		       stat_err == 0 ? 1 : 0);
		if (stat_err) {
			fprintf(stderr,
				"Start/end clock-offset intervals do not overlap\n");
			err = stat_err;
		} else {
			printf("Clock offset intersection lower: %" PRId64 " ns\n",
			       intersection_low);
			printf("Clock offset intersection upper: %" PRId64 " ns\n",
			       intersection_high);
		}
	}
	stat_err = print_histogram(skel);
	if (!err && stat_err)
		err = stat_err;

cleanup:
	free(func_name);
	/* Clean up */
	launchlate_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
