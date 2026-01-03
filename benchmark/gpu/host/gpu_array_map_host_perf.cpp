#include "bpftime_shm.hpp"
#include <cuda.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;

struct bench_options {
	uint64_t iters = 20000;
	uint64_t max_entries = 1024;
	uint64_t value_size = 8;
	bool enable_gdrcopy = false;
	uint64_t gdrcopy_max_per_key_bytes =
		DEFAULT_GPU_GDRCOPY_MAX_PER_KEY_BYTES;
};

static bool parse_u64_arg(const char *arg, const char *name, uint64_t &out)
{
	auto prefix = std::string("--") + name + "=";
	if (std::strncmp(arg, prefix.c_str(), prefix.size()) != 0) {
		return false;
	}
	const char *val = arg + prefix.size();
	if (!*val) {
		return false;
	}
	char *end = nullptr;
	uint64_t parsed = std::strtoull(val, &end, 10);
	if (!end || *end != '\0') {
		return false;
	}
	out = parsed;
	return true;
}

static bench_options parse_args(int argc, char **argv)
{
	bench_options opt;
	for (int i = 1; i < argc; ++i) {
		if (parse_u64_arg(argv[i], "iters", opt.iters) ||
		    parse_u64_arg(argv[i], "max-entries", opt.max_entries) ||
		    parse_u64_arg(argv[i], "value-size", opt.value_size) ||
		    parse_u64_arg(argv[i], "gdrcopy-max-per-key-bytes",
				  opt.gdrcopy_max_per_key_bytes)) {
			continue;
		}
		if (std::strcmp(argv[i], "--gdrcopy") == 0 && i + 1 < argc) {
			opt.enable_gdrcopy =
				std::strtoull(argv[++i], nullptr, 10) != 0;
			continue;
		}
		if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
			opt.iters = std::strtoull(argv[++i], nullptr, 10);
			continue;
		}
		if (std::strcmp(argv[i], "--max-entries") == 0 &&
		    i + 1 < argc) {
			opt.max_entries =
				std::strtoull(argv[++i], nullptr, 10);
			continue;
		}
		if (std::strcmp(argv[i], "--value-size") == 0 &&
		    i + 1 < argc) {
			opt.value_size =
				std::strtoull(argv[++i], nullptr, 10);
			continue;
		}
		if (std::strcmp(argv[i], "--gdrcopy-max-per-key-bytes") == 0 &&
		    i + 1 < argc) {
			opt.gdrcopy_max_per_key_bytes =
				std::strtoull(argv[++i], nullptr, 10);
			continue;
		}
		std::cerr << "Unknown argument: " << argv[i] << "\n";
		std::exit(1);
	}
	return opt;
}

static uint32_t lcg_next(uint32_t &state, uint32_t modulus)
{
	state = state * 1664525u + 1013904223u;
	if (modulus == 0) {
		return state;
	}
	return state % modulus;
}

static void print_result(const char *label, uint64_t iters,
			 std::chrono::nanoseconds duration)
{
	double ns_per_op =
		static_cast<double>(duration.count()) / (double)iters;
	double ops_per_sec = 1e9 / ns_per_op;
	std::cout.setf(std::ios::fixed);
	std::cout.precision(1);
	std::cout << label << ": " << ns_per_op << " ns/op"
		  << " (" << ops_per_sec << " ops/s)\n";
}

int main(int argc, char **argv)
{
	auto opt = parse_args(argc, argv);

	if (auto err = cuInit(0); err != CUDA_SUCCESS) {
		std::cerr << "cuInit(0) failed with error " << (int)err
			  << "\n";
		return 1;
	}

	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	auto config = bpftime::bpftime_get_agent_config();
	config.enable_gpu_gdrcopy = opt.enable_gdrcopy;
	config.gpu_gdrcopy_max_per_key_bytes = opt.gdrcopy_max_per_key_bytes;
	bpftime::bpftime_set_agent_config(std::move(config));

	bpftime::bpf_map_attr attr{};
	attr.type =
		(int)bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP;
	attr.key_size = sizeof(uint32_t);
	attr.value_size = (uint32_t)opt.value_size;
	attr.max_ents = (uint32_t)opt.max_entries;

	int map_fd =
		bpftime_maps_create(-1, "gpu_array_map_host_perf", attr);
	if (map_fd < 0) {
		std::perror("bpftime_maps_create");
		return 1;
	}

	std::vector<uint8_t> value(opt.value_size, 0);

	// Pre-fill the map with some values to avoid ENOENT on lookups.
	for (uint32_t i = 0; i < opt.max_entries; ++i) {
		uint32_t key = i;
		if (bpftime_map_update_elem(map_fd, &key, value.data(), 0) !=
		    0) {
			std::perror("bpftime_map_update_elem (prefill)");
			return 1;
		}
	}

	// Update benchmark.
	uint32_t state = 0x12345678u;
	auto start_update = clock_type::now();
	for (uint64_t i = 0; i < opt.iters; ++i) {
		uint32_t key =
			lcg_next(state, (uint32_t)opt.max_entries);
		if (bpftime_map_update_elem(map_fd, &key, value.data(), 0) !=
		    0) {
			std::perror("bpftime_map_update_elem");
			return 1;
		}
	}
	auto end_update = clock_type::now();

	// Lookup benchmark.
	state = 0xdeadbeefu;
	auto start_lookup = clock_type::now();
	for (uint64_t i = 0; i < opt.iters; ++i) {
		uint32_t key =
			lcg_next(state, (uint32_t)opt.max_entries);
		auto *ptr = bpftime_map_lookup_elem(map_fd, &key);
		if (!ptr) {
			std::perror("bpftime_map_lookup_elem");
			return 1;
		}
	}
	auto end_lookup = clock_type::now();

	auto update_ns =
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			end_update - start_update);
	auto lookup_ns =
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			end_lookup - start_lookup);

	std::cout << "iters=" << opt.iters
		  << " max_entries=" << opt.max_entries
		  << " value_size=" << opt.value_size << " bytes"
		  << " gdrcopy=" << (opt.enable_gdrcopy ? 1 : 0)
		  << " gdrcopy_max_per_key_bytes="
		  << opt.gdrcopy_max_per_key_bytes << "\n";
	print_result("update", opt.iters, update_ns);
	print_result("lookup", opt.iters, lookup_ns);

	bpftime_destroy_global_shm();
	return 0;
}
