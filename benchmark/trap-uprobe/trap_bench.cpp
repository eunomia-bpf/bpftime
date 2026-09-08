// Trap backend microbenchmark
//
// Measures per-hit latency of uprobe, uretprobe, and uprobe+uretprobe
// through the trap_attach_impl API.  Runs natively on riscv64 and under
// qemu-user for cross-compiled builds.
#include <trap_uprobe_attach_impl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <time.h>

#if defined(__GNUC__) && !defined(__clang__)
#define BENCH_TARGET __attribute__((noinline, noclone, used))
#else
#define BENCH_TARGET __attribute__((noinline, used))
#endif

using namespace bpftime::attach::trap;

static uprobe_callback make_noop()
{
	return [](const bpftime::pt_regs &) {};
}

// -----------------------------------------------------------------
// Target functions — one per benchmark so attaches don't interfere.
// -----------------------------------------------------------------
extern "C" BENCH_TARGET uint64_t __trap_bench_baseline(char *a, int b,
						       uint64_t c)
{
	asm("");
	return a[b] + c;
}
extern "C" BENCH_TARGET uint64_t __trap_bench_uprobe(char *a, int b,
						     uint64_t c)
{
	asm("");
	return a[b] + c;
}
extern "C" BENCH_TARGET uint64_t __trap_bench_uretprobe(char *a, int b,
							uint64_t c)
{
	asm("");
	return a[b] + c;
}
extern "C" BENCH_TARGET uint64_t __trap_bench_both(char *a, int b,
						   uint64_t c)
{
	asm("");
	return a[b] + c;
}

// -----------------------------------------------------------------
// Timing helpers
// -----------------------------------------------------------------
using bench_fn_t = uint64_t (*)(char *, int, uint64_t);

static double measure_ns_per_call(bench_fn_t fn, int iter)
{
	char buf[20] = "hello world";
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (int i = 0; i < iter; i++)
		fn(buf, i % 4, i);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double elapsed = (t1.tv_sec - t0.tv_sec) * 1e9 +
			 (t1.tv_nsec - t0.tv_nsec);
	return elapsed / iter;
}

// -----------------------------------------------------------------
// Multi-thread benchmark state
// -----------------------------------------------------------------
struct thread_ctx {
	bench_fn_t fn;
	int iter;
	int id;
	double ns_per_call;
};

static void *thread_entry(void *arg)
{
	auto *ctx = (thread_ctx *)arg;
	trap_attach_impl::prepare_thread();
	ctx->ns_per_call = measure_ns_per_call(ctx->fn, ctx->iter);
	return nullptr;
}

// -----------------------------------------------------------------
// main
// -----------------------------------------------------------------
int main(int argc, char **argv)
{
	int iter = 100000;
	int nthreads = 1;
	if (argc > 1)
		iter = atoi(argv[1]);
	if (argc > 2)
		nthreads = atoi(argv[2]);

	trap_attach_impl man;

	// --- baseline (no probe) ---
	double baseline = measure_ns_per_call(__trap_bench_baseline, iter);
	printf("Benchmarking baseline (no probe) in thread 1\n");
	printf("Average time usage %lf ns, iter %d times\n\n", baseline, iter);

	// --- uprobe only ---
	int uid = man.create_uprobe_at((void *)__trap_bench_uprobe,
				       make_noop());
	if (uid < 0) {
		fprintf(stderr, "create_uprobe_at failed: %d\n", uid);
		return 1;
	}
	double uprobe_ns = measure_ns_per_call(__trap_bench_uprobe, iter);
	printf("Benchmarking __trap_bench_uprobe in thread 1\n");
	printf("Average time usage %lf ns, iter %d times\n\n", uprobe_ns,
	       iter);

	// --- uretprobe only ---
	int rid = man.create_uretprobe_at((void *)__trap_bench_uretprobe,
					  make_noop());
	if (rid < 0) {
		fprintf(stderr, "create_uretprobe_at failed: %d\n", rid);
		return 1;
	}
	double uretprobe_ns =
		measure_ns_per_call(__trap_bench_uretprobe, iter);
	printf("Benchmarking __trap_bench_uretprobe in thread 1\n");
	printf("Average time usage %lf ns, iter %d times\n\n", uretprobe_ns,
	       iter);

	// --- uprobe + uretprobe ---
	int bid1 = man.create_uprobe_at((void *)__trap_bench_both,
					make_noop());
	int bid2 = man.create_uretprobe_at((void *)__trap_bench_both,
					   make_noop());
	if (bid1 < 0 || bid2 < 0) {
		fprintf(stderr, "create uprobe+uretprobe failed\n");
		return 1;
	}
	double both_ns = measure_ns_per_call(__trap_bench_both, iter);
	printf("Benchmarking __trap_bench_uprobe_uretprobe in thread 1\n");
	printf("Average time usage %lf ns, iter %d times\n\n", both_ns, iter);

	// --- multi-thread (uprobe + uretprobe on __trap_bench_both) ---
	if (nthreads > 1) {
		printf("--- Multi-thread: %d threads, uprobe+uretprobe ---\n\n",
		       nthreads);
		pthread_t threads[nthreads];
		thread_ctx ctxs[nthreads];
		for (int i = 0; i < nthreads; i++) {
			ctxs[i] = { __trap_bench_both, iter, i + 1, 0.0 };
			pthread_create(&threads[i], nullptr, thread_entry,
				       &ctxs[i]);
		}
		for (int i = 0; i < nthreads; i++) {
			pthread_join(threads[i], nullptr);
			printf("Thread %d: Average time usage %lf ns, iter %d times\n",
			       ctxs[i].id, ctxs[i].ns_per_call, iter);
		}
		printf("\n");
	}

	// --- summary ---
	printf("=== Summary (ns/call) ===\n");
	printf("%-30s %12.2f\n", "Baseline (no probe)", baseline);
	printf("%-30s %12.2f\n", "Uprobe only", uprobe_ns);
	printf("%-30s %12.2f\n", "Uretprobe only", uretprobe_ns);
	printf("%-30s %12.2f\n", "Uprobe + Uretprobe", both_ns);
	printf("%-30s %12.2f\n", "Uprobe overhead",
	       uprobe_ns - baseline);
	printf("%-30s %12.2f\n", "Uretprobe overhead",
	       uretprobe_ns - baseline);
	printf("%-30s %12.2f\n", "Both overhead", both_ns - baseline);

	return 0;
}
