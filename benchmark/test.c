#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

#define BENCH_FUNC(name) \
__attribute_noinline__ uint64_t name(char *a, int b, uint64_t c) \
{ \
    return a[b] + c; \
}

BENCH_FUNC(__bench_array_map_lookup)
BENCH_FUNC(__bench_array_map_delete)
BENCH_FUNC(__bench_array_map_update)
BENCH_FUNC(__bench_hash_map_lookup)
BENCH_FUNC(__bench_hash_map_delete)
BENCH_FUNC(__bench_hash_map_update)
BENCH_FUNC(__bench_per_cpu_hash_map_lookup)
BENCH_FUNC(__bench_per_cpu_hash_map_delete)
BENCH_FUNC(__bench_per_cpu_hash_map_update)
BENCH_FUNC(__bench_per_cpu_array_map_lookup)
BENCH_FUNC(__bench_per_cpu_array_map_delete)
BENCH_FUNC(__bench_per_cpu_array_map_update)
BENCH_FUNC(__bench_read)
BENCH_FUNC(__bench_write)
BENCH_FUNC(__bench_uprobe)
BENCH_FUNC(__bench_uretprobe)
BENCH_FUNC(__bench_uprobe_uretprobe)

typedef uint64_t (*benchmark_test_function_t)(char *, int, uint64_t);

void start_timer(struct timespec *start_time)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, start_time);
}

void end_timer(struct timespec *end_time)
{
	clock_gettime(CLOCK_MONOTONIC_RAW, end_time);
}

static double get_elapsed_time(struct timespec start_time,
			       struct timespec end_time)
{
	long seconds = end_time.tv_sec - start_time.tv_sec;
	long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
	if (start_time.tv_nsec > end_time.tv_nsec) { // clock underflow
		--seconds;
		nanoseconds += 1000000000;
	}
	return seconds * 1.0 + nanoseconds / 1000000000.0;
}

static double get_function_time(benchmark_test_function_t func, int iter)
{
	// The timespec struct holds seconds and nanoseconds
	struct timespec start_time, end_time;
	start_timer(&start_time);
	char buffer[20] = "hello world";
	// test base line
	for (int i = 0; i < iter; i++) {
		func(buffer, i % 4, i);
	}
	end_timer(&end_time);
	double time = get_elapsed_time(start_time, end_time);
	return time;
}

void do_benchmark_userspace(benchmark_test_function_t func, const char *name,
			    int iter, int id)
{
	double base_line_time, after_hook_time, total_time;

	base_line_time = get_function_time(func, iter);
	printf("Benchmarking %s in thread %d\nAverage time usage %lf ns, iter %d times\n\n",
	       name, id, (base_line_time) / iter * 1000000000.0, iter);
}

#define do_benchmark_func(func, iter, id)                                      \
	do {                                                                   \
		do_benchmark_userspace(func, #func, iter, id);                 \
	} while (0)

int iter = 100 * 1000;

void *run_bench_functions(void *id_ptr)
{
	int id = *(int *)id_ptr;
	printf("id: %d\n", id);
	do_benchmark_func(__bench_uprobe_uretprobe, iter, id);
	do_benchmark_func(__bench_uretprobe, iter, id);
	do_benchmark_func(__bench_uprobe, iter, id);
	do_benchmark_func(__bench_read, iter, id);
	do_benchmark_func(__bench_write, iter, id);
	do_benchmark_func(__bench_hash_map_update, iter, id);
	do_benchmark_func(__bench_hash_map_lookup, iter, id);
	do_benchmark_func(__bench_hash_map_delete, iter, id);
	do_benchmark_func(__bench_array_map_update, iter, id);
	do_benchmark_func(__bench_array_map_lookup, iter, id);
	do_benchmark_func(__bench_array_map_delete, iter, id);
	do_benchmark_func(__bench_per_cpu_hash_map_update, iter, id);
	do_benchmark_func(__bench_per_cpu_hash_map_lookup, iter, id);
	do_benchmark_func(__bench_per_cpu_hash_map_delete, iter, id);
	do_benchmark_func(__bench_per_cpu_array_map_update, iter, id);
	do_benchmark_func(__bench_per_cpu_array_map_lookup, iter, id);
	do_benchmark_func(__bench_per_cpu_array_map_delete, iter, id);
	return NULL;
}

int main(int argc, char **argv)
{
	int NUM_THREADS = 1;
	if (argc > 1) {
		NUM_THREADS = atoi(argv[1]);
	}
	if (argc > 2) {
		iter = atoi(argv[2]);
	}
	if (NUM_THREADS == 1) {
		run_bench_functions(&NUM_THREADS);
		return 0;
	}
	pthread_t threads[NUM_THREADS];
	int thread_id[NUM_THREADS];

	for (int i = 0; i < NUM_THREADS; i++) {
		thread_id[i] = i;
		pthread_create(&threads[i], NULL, run_bench_functions,
			       (void *)&thread_id[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
}
