#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_DRIVER_CHECK(expr)                                               \
	do {                                                                  \
		cudaError_t err__ = (expr);                                   \
		if (err__ != cudaSuccess) {                                   \
			std::cerr << "CUDA error " << cudaGetErrorString(err__)  \
				  << " at " << __FILE__ << ":" << __LINE__ \
				  << std::endl;                            \
			std::exit(EXIT_FAILURE);                                \
		}                                                             \
	} while (0)

namespace {
struct GemmOptions {
	int m = 4096;
	int n = 4096;
	int k = 4096;
	int launches = 24;
	int warmup = 2;
	unsigned int seed = 42;
	bool verify = false;
};

void usage(const char *prog)
{
	std::cerr
		<< "Usage: " << prog
		<< " [--shape MxNxK] [--m M] [--n N] [--k K] [--launches N] "
		   "[--warmup N] [--seed SEED] [--verify]" << std::endl
		<< std::endl
		<< "  Defaults target a heavy GEMM (4096^3, 24 launches). Use"
		   " --verify only for small matrices (<=512^3)." << std::endl
		<< std::endl;
}

int parse_positive_int(const std::string &value, const char *flag)
{
	int parsed = 0;
	try {
		size_t idx = 0;
		parsed = std::stoi(value, &idx);
		if (idx != value.size())
			throw std::invalid_argument("extra characters");
	} catch (const std::exception &ex) {
		std::cerr << "Failed to parse value for " << flag << ": "
			  << ex.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	if (parsed <= 0) {
		std::cerr << flag << " must be positive" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return parsed;
}

void parse_shape(const std::string &shape, GemmOptions &opts)
{
	const auto first_x = shape.find('x');
	const auto second_x = shape.find('x', first_x == std::string::npos ?
						    std::string::npos :
						    first_x + 1);
	if (first_x == std::string::npos || second_x == std::string::npos) {
		std::cerr << "Invalid --shape value (expected MxNxK): "
			  << shape << std::endl;
		std::exit(EXIT_FAILURE);
	}
	opts.m = parse_positive_int(shape.substr(0, first_x), "--shape");
	opts.n = parse_positive_int(
		shape.substr(first_x + 1, second_x - first_x - 1), "--shape");
	opts.k = parse_positive_int(shape.substr(second_x + 1), "--shape");
}

GemmOptions parse_options(int argc, char **argv)
{
	GemmOptions opts;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		auto require_value = [&](const char *flag) -> std::string {
			if (i + 1 >= argc) {
				std::cerr << flag << " expects a value"
					  << std::endl;
				std::exit(EXIT_FAILURE);
			}
			return argv[++i];
		};
		if (arg == "--m") {
			opts.m = parse_positive_int(require_value("--m"),
						    "--m");
		} else if (arg == "--n") {
			opts.n = parse_positive_int(require_value("--n"),
						    "--n");
		} else if (arg == "--k") {
			opts.k = parse_positive_int(require_value("--k"),
						    "--k");
		} else if (arg == "--shape") {
			parse_shape(require_value("--shape"), opts);
		} else if (arg.rfind("--shape=", 0) == 0) {
			parse_shape(arg.substr(8), opts);
		} else if (arg == "--launches") {
			opts.launches = parse_positive_int(
				require_value("--launches"), "--launches");
		} else if (arg == "--warmup") {
			opts.warmup = parse_positive_int(
				require_value("--warmup"), "--warmup");
		} else if (arg == "--seed") {
			opts.seed =
				static_cast<unsigned int>(parse_positive_int(
					require_value("--seed"), "--seed"));
		} else if (arg == "--verify") {
			opts.verify = true;
		} else if (arg == "--help" || arg == "-h") {
			usage(argv[0]);
			std::exit(EXIT_SUCCESS);
		} else {
			std::cerr << "Unknown argument: " << arg << std::endl;
			usage(argv[0]);
			std::exit(EXIT_FAILURE);
		}
	}
	return opts;
}

size_t bytes_for_elements(size_t elements)
{
	const size_t max_elements =
		static_cast<size_t>(std::numeric_limits<size_t>::max() /
				    sizeof(float));
	if (elements > max_elements) {
		std::cerr << "Requested allocation exceeds addressable limit"
			  << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return elements * sizeof(float);
}
} // namespace

int main(int argc, char **argv)
{
	const GemmOptions opts = parse_options(argc, argv);
	using Layout = cutlass::layout::RowMajor;
	using Gemm = cutlass::gemm::device::Gemm<float, Layout, float, Layout,
						 float, Layout>;

	const size_t elements_A =
		static_cast<size_t>(opts.m) * static_cast<size_t>(opts.k);
	const size_t elements_B =
		static_cast<size_t>(opts.k) * static_cast<size_t>(opts.n);
	const size_t elements_C =
		static_cast<size_t>(opts.m) * static_cast<size_t>(opts.n);

	std::cout << "[cutlass] problem: " << opts.m << "x" << opts.n << "x"
		  << opts.k << " | launches=" << opts.launches
		  << " (warmup=" << opts.warmup << ")"
		  << " | host seed=" << opts.seed << std::endl;

	std::vector<float> host_A(elements_A);
	std::vector<float> host_B(elements_B);
	std::vector<float> host_C(elements_C, 0.0f);
	std::vector<float> host_D(elements_C, 0.0f);

	std::mt19937 rng(opts.seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto &v : host_A)
		v = dist(rng);
	for (auto &v : host_B)
		v = dist(rng);

	float *d_A = nullptr;
	float *d_B = nullptr;
	float *d_C = nullptr;
	float *d_D = nullptr;
	CUDA_DRIVER_CHECK(cudaMalloc(&d_A, bytes_for_elements(elements_A)));
	CUDA_DRIVER_CHECK(cudaMalloc(&d_B, bytes_for_elements(elements_B)));
	CUDA_DRIVER_CHECK(cudaMalloc(&d_C, bytes_for_elements(elements_C)));
	CUDA_DRIVER_CHECK(cudaMalloc(&d_D, bytes_for_elements(elements_C)));

	CUDA_DRIVER_CHECK(cudaMemcpy(d_A, host_A.data(),
				     bytes_for_elements(elements_A),
				     cudaMemcpyHostToDevice));
	CUDA_DRIVER_CHECK(cudaMemcpy(d_B, host_B.data(),
				     bytes_for_elements(elements_B),
				     cudaMemcpyHostToDevice));
	CUDA_DRIVER_CHECK(cudaMemcpy(d_C, host_C.data(),
				     bytes_for_elements(elements_C),
				     cudaMemcpyHostToDevice));

	cutlass::gemm::GemmCoord problem_size(opts.m, opts.n, opts.k);

	Layout layout_A(opts.k);
	Layout layout_B(opts.n);
	Layout layout_C(opts.n);

	cutlass::TensorRef<float, Layout> ref_A(d_A, layout_A);
	cutlass::TensorRef<float, Layout> ref_B(d_B, layout_B);
	cutlass::TensorRef<float, Layout> ref_C(d_C, layout_C);
	cutlass::TensorRef<float, Layout> ref_D(d_D, layout_C);

	Gemm gemm_op;
	Gemm::Arguments args(problem_size, ref_A, ref_B, ref_C, ref_D,
			     { 1.0f, 0.0f });

	auto run_gemm = [&](int count) -> bool {
		for (int launch = 0; launch < count; ++launch) {
			auto status = gemm_op(args);
			if (status != cutlass::Status::kSuccess) {
				std::cerr << "CUTLASS GEMM launch failed with "
					     "status "
					  << static_cast<int>(status)
					  << " on launch " << launch
					  << std::endl;
				return false;
			}
		}
		return true;
	};

	if (opts.warmup > 0) {
		std::cout << "[cutlass] running " << opts.warmup
			  << " warmup launches..." << std::endl;
		if (!run_gemm(opts.warmup))
			return EXIT_FAILURE;
		CUDA_DRIVER_CHECK(cudaDeviceSynchronize());
	}

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	CUDA_DRIVER_CHECK(cudaEventCreate(&start));
	CUDA_DRIVER_CHECK(cudaEventCreate(&stop));

	std::cout << "[cutlass] running timed workload..." << std::endl;
	CUDA_DRIVER_CHECK(cudaEventRecord(start));
	if (!run_gemm(opts.launches)) {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return EXIT_FAILURE;
	}
	CUDA_DRIVER_CHECK(cudaEventRecord(stop));
	CUDA_DRIVER_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms = 0.0f;
	CUDA_DRIVER_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	const double total_flops =
		2.0 * static_cast<double>(opts.m) * opts.n * opts.k;
	const double seconds = elapsed_ms / 1000.0;
	const double gflops =
		(seconds > 0.0) ? (total_flops * opts.launches / seconds /
				   1.0e9) :
				  0.0;
	std::cout << "[cutlass] completed " << opts.launches
		  << " launches in " << elapsed_ms << " ms "
		  << "(avg " << (elapsed_ms / opts.launches) << " ms) | "
		  << gflops << " GFLOP/s" << std::endl;

	CUDA_DRIVER_CHECK(cudaMemcpy(host_D.data(), d_D,
				     bytes_for_elements(elements_C),
				     cudaMemcpyDeviceToHost));

	double checksum = 0.0;
	double l2norm = 0.0;
	for (const float value : host_D) {
		checksum += static_cast<double>(value);
		l2norm += static_cast<double>(value) *
			  static_cast<double>(value);
	}
	std::cout << "[cutlass] checksum=" << checksum
		  << " | l2-norm=" << std::sqrt(l2norm) << std::endl;

	constexpr int kVerifyLimit = 512;
	if (opts.verify) {
		if (opts.m > kVerifyLimit || opts.n > kVerifyLimit ||
		    opts.k > kVerifyLimit) {
			std::cout
				<< "[cutlass] skipping CPU verification "
				   "(too large for "
				<< kVerifyLimit << "^3 limit)" << std::endl;
		} else {
			std::cout << "[cutlass] verifying on CPU..." << std::endl;
			std::vector<float> reference(host_D.size(), 0.0f);
			for (int m = 0; m < opts.m; ++m) {
				for (int n = 0; n < opts.n; ++n) {
					float accum = 0.0f;
					for (int k = 0; k < opts.k; ++k) {
						accum += host_A[m * opts.k + k] *
							host_B[k * opts.n + n];
					}
					reference[m * opts.n + n] = accum;
				}
			}
			double max_error = 0.0;
			for (size_t idx = 0; idx < reference.size(); ++idx) {
				max_error = std::max(
					max_error,
					std::abs(static_cast<double>(
							 reference[idx]) -
						 static_cast<double>(
							 host_D[idx])));
			}
			std::cout << "[cutlass] max abs error: " << max_error
				  << std::endl;
		}
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	return 0;
}
