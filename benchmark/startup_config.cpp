// Compare the former live-environment lookup with the local POD read.
// clang++ -std=c++20 -O2 -pthread -Iruntime/include \
//   benchmark/startup_config.cpp runtime/src/bpftime_config.cpp -o config-bench
#include "bpftime_config.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern "C" char **environ;
#endif

int main()
{
#if defined(__APPLE__)
	char **&environment = *_NSGetEnviron();
#else
	char **&environment = environ;
#endif
	char **original = environment;
	std::vector<std::string> values;
	for (int i = 0; i < 199; ++i)
		values.push_back("CONFIG_BENCH_" + std::to_string(i) +
				 "=value");
	values.emplace_back("BPFTIME_RUN_WITH_KERNEL=1");
	std::vector<char *> entries;
	for (auto &value : values)
		entries.push_back(value.data());
	entries.push_back(nullptr);
	environment = entries.data();
	bpftime::initialize_local_runtime_config_from_env();
	constexpr size_t iterations = 1000000;
	std::printf(
		"mode,threads,samples,iterations,mean_ns,median_ns,min_ns,max_ns\n");
	for (int count : { 1, 4 }) {
		for (bool local : { false, true }) {
			std::vector<double> samples;
			for (int run = 0; run < 7; ++run) {
				std::vector<double> elapsed(count);
				std::vector<std::thread> threads;
				for (int t = 0; t < count; ++t)
					threads.emplace_back([&, t] {
						size_t hits = 0;
						const auto start = std::chrono::
							steady_clock::now();
						for (size_t i = 0;
						     i < iterations; ++i) {
							// Force a load each
							// iteration without an
							// emitted fence.
							asm volatile(
								"" ::
									: "memory");
							if (local)
								hits += bpftime::bpftime_get_local_runtime_config()
										.run_with_kernel;
							else
								hits += getenv("BPFTIME_RUN_WITH_KERNEL") !=
									nullptr;
						}
						const auto stop = std::chrono::
							steady_clock::now();
						if (hits != iterations)
							std::abort();
						elapsed[t] =
							std::chrono::duration<
								double,
								std::nano>(
								stop - start)
								.count() /
							iterations;
					});
				for (auto &thread : threads)
					thread.join();
				double sum = 0;
				for (double time : elapsed)
					sum += time;
				samples.push_back(sum / count);
			}
			std::sort(samples.begin(), samples.end());
			double sum = 0;
			for (double sample : samples)
				sum += sample;
			std::printf("%s,%d,7,%zu,%.3f,%.3f,%.3f,%.3f\n",
				    local ? "local" : "master", count,
				    iterations, sum / samples.size(),
				    samples[3], samples.front(),
				    samples.back());
		}
	}
	environment = original;
}
