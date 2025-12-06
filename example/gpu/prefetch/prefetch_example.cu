#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include "synthetic.cuh"

struct Config {
	std::string kernel = "seq_stream";
	std::string mode = "uvm";
	float size_factor = 0.25; // Reduced default for lighter workload
	int iterations = 5; // Reduced default for faster runs
	size_t stride_bytes = 4096; // Access granularity (4096 = page-level for
				    // oversub tests)
	std::string output = "results.csv";
};

void print_usage(const char *prog)
{
	std::cout
		<< "Usage: " << prog << " [options]\n"
		<< "Options:\n"
		<< "  --kernel=<name>        Kernel: seq_stream, rand_stream, pointer_chase (default: seq_stream)\n"
		<< "  --mode=<mode>          Mode: device, uvm, uvm_prefetch, uvm_advise_read, \n"
		<< "                         uvm_advise_pref_gpu, uvm_advise_pref_cpu,\n"
		<< "                         uvm_advise_access (default: uvm)\n"
		<< "  --size_factor=<float>  Size factor relative to GPU memory (default: 0.25)\n"
		<< "                         device mode: requires <= 0.8\n"
		<< "                         uvm mode: can be > 1.0 for oversubscription\n"
		<< "  --stride_bytes=<int>   Access stride in bytes (default: 4096)\n"
		<< "                         4096 = page-level access (one element per page)\n"
		<< "                         4    = element-level access (all elements)\n"
		<< "  --iterations=<int>     Number of iterations (default: 5)\n"
		<< "  --output=<path>        Output CSV file (default: results.csv)\n"
		<< "  --help                 Show this help\n";
}

Config parse_args(int argc, char **argv)
{
	Config config;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "--help" || arg == "-h") {
			print_usage(argv[0]);
			exit(0);
		}

		if (arg.substr(0, 9) == "--kernel=") {
			config.kernel = arg.substr(9);
		} else if (arg.substr(0, 7) == "--mode=") {
			config.mode = arg.substr(7);
		} else if (arg.substr(0, 14) == "--size_factor=") {
			config.size_factor = std::stof(arg.substr(14));
		} else if (arg.substr(0, 15) == "--stride_bytes=") {
			config.stride_bytes = std::stoul(arg.substr(15));
		} else if (arg.substr(0, 13) == "--iterations=") {
			config.iterations = std::stoi(arg.substr(13));
		} else if (arg.substr(0, 9) == "--output=") {
			config.output = arg.substr(9);
		} else {
			std::cerr << "Unknown argument: " << arg << std::endl;
			print_usage(argv[0]);
			exit(1);
		}
	}

	// Validate device mode size_factor
	if (config.mode == "device" && config.size_factor > 0.8f) {
		std::cerr
			<< "Error: device mode requires size_factor <= 0.8 to avoid OOM\n";
		std::cerr
			<< "       (current size_factor: " << config.size_factor
			<< ")\n";
		std::cerr
			<< "       Use --size_factor=0.8 or smaller, or use --mode=uvm for oversubscription\n";
		exit(1);
	}

	return config;
}

size_t get_gpu_memory()
{
	size_t free_bytes, total_bytes;
	cudaMemGetInfo(&free_bytes, &total_bytes);
	return total_bytes;
}

// ============================================================================
// Kernel Registry System
// ============================================================================

using KernelRunner = void (*)(size_t, const std::string &, size_t, int,
			      std::vector<float> &, KernelResult &);

struct KernelEntry {
	const char *name;
	const char *description;
	KernelRunner runner;
};

// Registry of available kernels
const KernelEntry g_kernels[] = {
	{ "seq_stream", "Sequential streaming with light compute",
	  run_seq_stream },
	{ "seq_device_prefetch",
	  "Sequential with GPU-side PTX prefetch.global.L2",
	  run_seq_device_prefetch },
	{ "rand_stream", "Random access pattern with index indirection",
	  run_rand_stream },
	{ "pointer_chase", "Pointer chasing for TLB/cache stress",
	  run_pointer_chase },
	// Add new kernels here: {"gemm", "Matrix multiply", run_gemm},
};

const int g_num_kernels = sizeof(g_kernels) / sizeof(g_kernels[0]);

KernelRunner lookup_kernel(const std::string &name)
{
	for (int i = 0; i < g_num_kernels; ++i) {
		if (name == g_kernels[i].name) {
			return g_kernels[i].runner;
		}
	}
	return nullptr;
}

void print_available_kernels()
{
	std::cout << "Available kernels:\n";
	for (int i = 0; i < g_num_kernels; ++i) {
		std::cout << "  " << g_kernels[i].name << " - "
			  << g_kernels[i].description << "\n";
	}
}

void write_results(const Config &config, const KernelResult &result,
		   size_t total_working_set)
{
	FILE *f = fopen(config.output.c_str(), "w");
	if (!f) {
		std::cerr << "Failed to open output file: " << config.output
			  << std::endl;
		return;
	}

	// Calculate bandwidth
	double bw_GBps =
		(result.bytes_accessed / (result.median_ms / 1000.0)) / 1e9;

	// CSV header
	fprintf(f,
		"kernel,mode,size_factor,working_set_bytes,bytes_accessed,iterations,"
		"median_ms,min_ms,max_ms,bw_GBps\n");

	// Write data
	fprintf(f, "%s,%s,%.2f,%zu,%zu,%d,%.3f,%.3f,%.3f,%.3f\n",
		config.kernel.c_str(), config.mode.c_str(), config.size_factor,
		total_working_set, result.bytes_accessed, config.iterations,
		result.median_ms, result.min_ms, result.max_ms, bw_GBps);

	fclose(f);

	std::cout
		<< "\nResults:\n"
		<< "  Kernel: " << config.kernel << "\n"
		<< "  Mode: " << config.mode << "\n"
		<< "  Working Set: " << total_working_set / (1024 * 1024)
		<< " MB\n"
		<< "  Bytes Accessed: " << result.bytes_accessed / (1024 * 1024)
		<< " MB\n"
		<< "  Median time: " << result.median_ms << " ms\n"
		<< "  Min time: " << result.min_ms << " ms\n"
		<< "  Max time: " << result.max_ms << " ms\n"
		<< "  Bandwidth: " << bw_GBps << " GB/s\n"
		<< "  Results written to: " << config.output << "\n";
}

int main(int argc, char **argv)
{
	Config config = parse_args(argc, argv);

	// Initialize CUDA with error checking
	try {
		CUDA_CHECK(cudaSetDevice(0));
	} catch (const std::exception &e) {
		std::cerr << "Failed to initialize CUDA: " << e.what()
			  << std::endl;
		return 1;
	}

	// Get GPU memory and compute size
	size_t gpu_mem = get_gpu_memory();
	size_t total_working_set =
		static_cast<size_t>(gpu_mem * config.size_factor);

	std::cout << "UVM Microbenchmark - Tier 0 Synthetic Kernels\n"
		  << "==============================================\n"
		  << "GPU Memory: " << gpu_mem / (1024 * 1024) << " MB\n"
		  << "Size Factor: " << config.size_factor;
	if (config.size_factor > 1.0f) {
		std::cout << " (oversubscription)";
	}
	std::cout << "\n"
		  << "Total Working Set: " << total_working_set / (1024 * 1024)
		  << " MB\n"
		  << "Stride Bytes: " << config.stride_bytes;
	if (config.stride_bytes == 4096) {
		std::cout << " (page-level)";
	} else if (config.stride_bytes == sizeof(float)) {
		std::cout << " (element-level)";
	}
	std::cout << "\n"
		  << "Kernel: " << config.kernel << "\n"
		  << "Mode: " << config.mode << "\n"
		  << "Iterations: " << config.iterations << "\n\n";

	// Run the selected kernel using registry
	std::vector<float> runtimes;
	KernelResult result;

	try {
		KernelRunner runner = lookup_kernel(config.kernel);
		if (!runner) {
			std::cerr << "Error: Unknown kernel '" << config.kernel
				  << "'\n\n";
			print_available_kernels();
			return 1;
		}

		runner(total_working_set, config.mode, config.stride_bytes,
		       config.iterations, runtimes, result);

		// Write results
		write_results(config, result, total_working_set);

	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
