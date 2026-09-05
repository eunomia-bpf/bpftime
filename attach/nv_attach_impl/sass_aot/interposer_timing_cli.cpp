// Repeated uninstrumented-vs-interposed timing CLI.
//
// Runs the PTX-free fixture application `runs` times without the
// interposer and `runs` times with the first-party LD_PRELOAD CUDA Driver
// API interposer (launch-boundary interposition of the verified
// BPF-derived SASS callback), capturing every output line and every
// return code:
//
//   usage: bpftime_sass_aot_interpose_timing [runs] [iterations]
//                                            [device-ordinal]
//                                            [artifact-directory]
//
//   === run 1/3 uninstrumented rc=0 ===
//   iter 0 launch_ns 4123 sync_ns 187234 total_ns 191357
//   ...
//   --- uninstrumented run 1 cold launch_ns=4123 sync_ns=187234
//       total_ns=191357 steady launch_ns_sum=... sync_ns_sum=...
//       total_ns_sum=... rc=0 ---
//   === run 1/3 interposed rc=0 ===
//   [sass_aot_interposer] launch-boundary interposition for target 'companion_app_kernel'
//   iter 0 launch_ns 1234567 sync_ns ...
//   ...
//   --- interposed run 1 cold launch_ns=... ... rc=0 ---
//
// All numeric timings and return codes are preserved verbatim in the
// per-run blocks. The per-run summaries report the cold cost (iteration 0,
// which includes context/module setup and, for interposed runs, the one-time
// BPF-to-SASS compilation at the first launch) and the steady cost
// (iterations 1..N-1). There are no gates, retries, or filtering: every run
// and every iteration is reported as captured.

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sass_aot_interposer_config.hpp"
#include "sass_aot_test_config.hpp"

namespace
{

std::string capture_run(const std::vector<std::string> &args,
			std::string &output, int &status)
{
	// Run `args` with stdout+stderr merged into a captured stream.
	// Returns an error string, or empty on success.
	int pipe_fds[2];
	if (::pipe(pipe_fds) != 0)
		return std::string("pipe failed: ") + std::strerror(errno);

	const pid_t child = ::fork();
	if (child < 0) {
		::close(pipe_fds[0]);
		::close(pipe_fds[1]);
		return std::string("fork failed: ") + std::strerror(errno);
	}
	if (child == 0) {
		::close(pipe_fds[0]);
		if (::dup2(pipe_fds[1], STDOUT_FILENO) < 0 ||
		    ::dup2(pipe_fds[1], STDERR_FILENO) < 0)
			::_exit(126);
		::close(pipe_fds[1]);
		std::vector<char *> argv;
		argv.reserve(args.size() + 1);
		for (const auto &arg : args)
			argv.push_back(const_cast<char *>(arg.c_str()));
		argv.push_back(nullptr);
		::execv(argv[0], argv.data());
		::_exit(127);
	}
	::close(pipe_fds[1]);

	std::ostringstream captured;
	char buffer[4096];
	for (;;) {
		const ssize_t n = ::read(pipe_fds[0], buffer, sizeof(buffer));
		if (n > 0) {
			captured.write(buffer, n);
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		break;
	}
	::close(pipe_fds[0]);

	int wait_status = 0;
	pid_t waited = -1;
	do {
		waited = ::waitpid(child, &wait_status, 0);
	} while (waited < 0 && errno == EINTR);
	if (waited < 0)
		return "waitpid failed";
	if (WIFEXITED(wait_status))
		status = WEXITSTATUS(wait_status);
	else if (WIFSIGNALED(wait_status))
		status = 128 + WTERMSIG(wait_status);
	else
		status = -1;
	output = captured.str();
	return {};
}

struct RunTimings {
	// Iteration 0 (cold): context/module setup plus, for interposed
	// runs, the one-time BPF-to-SASS compilation at the first launch.
	bool has_cold = false;
	long long cold_launch_ns = 0;
	long long cold_sync_ns = 0;
	long long cold_total_ns = 0;
	// Iterations 1..N-1 (steady).
	long long steady_launch_ns_sum = 0;
	long long steady_sync_ns_sum = 0;
	long long steady_total_ns_sum = 0;
	int steady_iterations = 0;
};

// Parse the application's `iter <i> launch_ns <n> sync_ns <n> total_ns <n>`
// lines (all preserved verbatim in the run block) into cold/steady sums.
RunTimings parse_timings(const std::string &output)
{
	RunTimings timings;
	size_t pos = 0;
	for (;;) {
		const size_t line = output.find('\n', pos);
		const std::string slice =
			output.substr(pos, line == std::string::npos
					  ? std::string::npos
					  : line - pos);
		if (slice.rfind("iter ", 0) == 0) {
			char *end = nullptr;
			const long long iteration =
				std::strtoll(slice.c_str() + 5, &end, 10);
			auto field = [&](const char *name) -> long long {
				const size_t at = slice.rfind(name);
				if (at == std::string::npos)
					return 0;
				char *field_end = nullptr;
				return std::strtoll(
					slice.c_str() + at + std::strlen(name),
					&field_end, 10);
			};
			if (iteration == 0) {
				timings.cold_launch_ns = field("launch_ns ");
				timings.cold_sync_ns = field("sync_ns ");
				timings.cold_total_ns = field("total_ns ");
				timings.has_cold = true;
			} else {
				timings.steady_launch_ns_sum +=
					field("launch_ns ");
				timings.steady_sync_ns_sum += field("sync_ns ");
				timings.steady_total_ns_sum +=
					field("total_ns ");
				++timings.steady_iterations;
			}
		}
		if (line == std::string::npos)
			break;
		pos = line + 1;
	}
	return timings;
}

struct RunSpec {
	const char *label;
	bool interposed;
};

int run_once(const std::string &app_binary, const std::string &app_cubin,
	     int iterations, int device_ordinal,
	     const std::string &artifact_dir, const RunSpec &spec,
	     std::string &output, int &status)
{
	std::vector<std::string> args = {
		app_binary,
		app_cubin,
		std::to_string(iterations),
		std::to_string(device_ordinal),
	};

	// Every run starts from the same environment; the uninstrumented
	// run strips interposer variables, the interposed run sets them.
	const char *strip[] = {
		"LD_PRELOAD", "BPFTIME_SASS_AOT_TARGET",
		"BPFTIME_SASS_AOT_BPF_OBJECT", "BPFTIME_SASS_AOT_SECTION",
		"BPFTIME_SASS_AOT_OUT_DIR",
	};
	for (const char *name : strip)
		(void)unsetenv(name);

	if (spec.interposed) {
		if (setenv("LD_PRELOAD", SASS_AOT_INTERPOSER_LIB, 1) != 0 ||
		    setenv("BPFTIME_SASS_AOT_TARGET", "companion_app_kernel",
			   1) != 0 ||
		    setenv("BPFTIME_SASS_AOT_BPF_OBJECT",
			   SASS_AOT_SPIKE_BPF_OBJECT, 1) != 0 ||
		    setenv("BPFTIME_SASS_AOT_SECTION", "cuda__/sass_aot", 1) !=
			    0 ||
		    setenv("BPFTIME_SASS_AOT_OUT_DIR", artifact_dir.c_str(),
			   1) != 0)
			return -1;
	}

	std::string error = capture_run(args, output, status);
	if (!error.empty())
		std::cerr << "interpose_timing: " << error << " (" << spec.label
			  << ")\n";
	return error.empty() ? 0 : -1;
}

} // namespace

int main(int argc, char **argv)
{
	if (argc > 5) {
		std::cerr << "usage: " << argv[0]
			  << " [runs] [iterations] [device-ordinal] "
			     "[artifact-directory]\n";
		return 2;
	}

	auto parse_bounded = [](const char *text, int minimum,
				const char *what) -> int {
		char *end = nullptr;
		const long value = std::strtol(text, &end, 10);
		if (end == text || *end != '\0' || value < minimum) {
			std::cerr << what << " must be an integer >= "
				  << minimum << '\n';
			std::exit(2);
		}
		return static_cast<int>(value);
	};

	int runs = 1;
	int iterations = 4;
	int device_ordinal = 0;
	std::string artifact_dir = SASS_AOT_INTERPOSER_DEFAULT_ARTIFACT_DIR;
	if (argc >= 2)
		runs = parse_bounded(argv[1], 1, "runs");
	if (argc >= 3)
		iterations = parse_bounded(argv[2], 1, "iterations");
	if (argc >= 4)
		device_ordinal = parse_bounded(argv[3], 0, "device ordinal");
	if (argc >= 5)
		artifact_dir = argv[4];

	std::cout << "app: " << SASS_AOT_INTERPOSER_APP_BIN << '\n';
	std::cout << "cubin: " << SASS_AOT_COMPANION_APP_CUBIN << '\n';
	std::cout << "bpf object: " << SASS_AOT_SPIKE_BPF_OBJECT << '\n';
	std::cout << "interposer: " << SASS_AOT_INTERPOSER_LIB << '\n';
	std::cout << "artifact dir: " << artifact_dir << '\n';
	std::cout << "runs: " << runs << " iterations: " << iterations
		  << " device: " << device_ordinal << '\n';

	bool all_ok = true;
	for (int r = 1; r <= runs; ++r) {
		const RunSpec specs[] = {
			{ "uninstrumented", false },
			{ "interposed", true },
		};
		for (const auto &spec : specs) {
			std::string output;
			int status = -1;
			const int err =
				run_once(SASS_AOT_INTERPOSER_APP_BIN,
					 SASS_AOT_COMPANION_APP_CUBIN,
					 iterations, device_ordinal,
					 artifact_dir, spec, output, status);
			std::cout << "=== run " << r << "/" << runs << ' '
				  << spec.label << " rc=" << status << " ===\n";
			std::cout << output;
			if (!output.empty() &&
			    output.back() != '\n')
				std::cout << '\n';
			if (err != 0 || status != 0)
				all_ok = false;
			const RunTimings timings = parse_timings(output);
			std::cout << "--- " << spec.label << " run " << r;
			if (timings.has_cold) {
				std::cout << " cold launch_ns="
					  << timings.cold_launch_ns
					  << " sync_ns=" << timings.cold_sync_ns
					  << " total_ns="
					  << timings.cold_total_ns;
			}
			std::cout << " steady launch_ns_sum="
				  << timings.steady_launch_ns_sum
				  << " sync_ns_sum="
				  << timings.steady_sync_ns_sum
				  << " total_ns_sum="
				  << timings.steady_total_ns_sum
				  << " (" << timings.steady_iterations
				  << " iterations) rc=" << status << " ---\n";
		}
	}

	std::cout << "repeated uninstrumented-vs-interposed timing "
		  << (all_ok ? "complete" : "completed with failures")
		  << '\n';
	return all_ok ? 0 : 1;
}
