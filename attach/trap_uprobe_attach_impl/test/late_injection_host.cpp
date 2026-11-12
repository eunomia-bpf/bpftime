#include "trap_test_common.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>
#include <vector>

extern "C" TRAP_TEST_TARGET uint64_t late_target(uint64_t value, uint64_t worker)
{
	asm volatile("" : : "r"(worker));
	return value + 1;
}

int main(int argc, char **argv)
{
	if (argc != 2)
		return 1;
	std::atomic<bool> stop{ false };
	std::atomic<unsigned> ready{ 0 }, errors{ 0 };
	std::atomic<unsigned> hits[5]{};
	std::vector<std::thread> workers;
	for (int i = 0; i < 4; ++i) {
		workers.emplace_back([&, i] {
			ready.fetch_add(1);
			while (!stop.load()) {
				void *p = malloc(128);
				asm volatile("" : : "r"(p) : "memory");
				if (late_target(41, i) != 42)
					errors.fetch_add(1);
				free(p);
			}
		});
	}
	while (ready.load() != 4)
		std::this_thread::yield();
	// The executable does not link bpftime. All workers exist before dlopen,
	// and no worker calls prepare_thread or any DSO function before its trap.
	void *dso = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	int result = 1;
	if (dso) {
		auto install = (int (*)(void *, std::atomic<unsigned> *))dlsym(dso, "install_probes");
		auto remove = (int (*)())dlsym(dso, "remove_probes");
		auto unsafe = (unsigned (*)())dlsym(dso, "unsafe_operation_count");
		if (install && remove && unsafe && install((void *)late_target, hits) == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			stop.store(true);
			for (auto &worker : workers)
				worker.join();
			const int detached = remove();
			const unsigned bad_calls = unsafe();
			printf("late injection: hits=%u errors=%u unsafe operations=%u\n",
			       hits[0].load(), errors.load(), bad_calls);
			bool all_workers_hit = true;
			for (int i = 1; i <= 4; ++i)
				all_workers_hit &= hits[i].load() > 0;
			result = detached == 0 && all_workers_hit && hits[0].load() > 0 && errors.load() == 0 && bad_calls == 0 ? 0 : 1;
		} else {
			stop.store(true);
		}
	} else {
		fprintf(stderr, "late dlopen failed: %s\n", dlerror());
	}
	stop.store(true);
	for (auto &worker : workers)
		if (worker.joinable())
			worker.join();
	// Keep the DSO resident if partial installation failed: it can still own
	// a breakpoint or return trampoline. Process exit reclaims the test DSO.
	if (dso && result == 0)
		dlclose(dso);
	return result;
}
