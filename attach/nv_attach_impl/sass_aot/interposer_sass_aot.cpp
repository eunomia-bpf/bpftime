// First-party LD_PRELOAD CUDA Driver API interposer: launch-boundary
// interposition for the verified BPF-derived SASS callback.
//
// This is launch-boundary interposition, not in-body SASS rewriting: the
// application's own modules, cubin, and SASS are never modified. The
// interposer records CUfunction names at cuModuleGetFunction and, for the
// configured target kernel, executes the verified BPF-derived SASS callback
// from within cuLaunchKernel (through the existing
// bpftime::attach::sass_aot::execute_sass_aot_in_context boundary) before
// launching the original application kernel. The application itself never
// calls execute_sass_aot_in_context.
//
// Reentrancy safety: because this library shadows cuModuleGetFunction and
// cuLaunchKernel process-wide (it sits first in the global symbol scope),
// the callback's internal CUDA calls - execute_sass_aot_in_context calls
// cuModuleGetFunction and cuLaunchKernel for the BPF-derived entry -
// re-enter this interposer on the same thread. Two rules make that safe:
//  - g_mutex is never held across a CUDA call or the one-time compilation
//    (verifier + ptxas fork/exec); it only guards the function-name map and
//    the pipeline state.
//  - a thread-local guard (g_interposing) marks the callback's own CUDA
//    calls so they pass through without bookkeeping or instrumentation and
//    never take g_mutex.
//
// Environment configuration (all optional; unset target => fully
// transparent pass-through):
//   BPFTIME_SASS_AOT_TARGET      kernel name to interpose (default: unset)
//   BPFTIME_SASS_AOT_BPF_OBJECT  BPF ELF object path
//   BPFTIME_SASS_AOT_SECTION     BPF program section (default cuda__/sass_aot)
//   BPFTIME_SASS_AOT_OUT_DIR     directory receiving sass_aot.ptx/cubin

#include <cuda.h>

#include <dlfcn.h>
#include <condition_variable>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sass_aot.hpp"

namespace
{

using RealModuleGetFunction =
	CUresult (*)(CUfunction *, CUmodule, const char *);
using RealLaunchKernel = CUresult (*)(CUfunction, unsigned int, unsigned int,
				       unsigned int, unsigned int, unsigned int,
				       unsigned int, unsigned int, CUstream,
				       void **, void **);

std::mutex g_mutex;
// Guards g_function_names, g_target/g_bpf_object/g_bpf_section/g_out_dir,
// g_pipeline, and g_compiled. Never held across a CUDA call or the
// one-time BPF-to-SASS compilation.
std::unordered_map<CUfunction, std::string> g_function_names;

std::string g_target;
std::string g_bpf_object;
std::string g_bpf_section = "cuda__/sass_aot";
std::string g_out_dir;
bool g_config_read = false;

enum class PipelineState
{
	uninitialized,
	compiling,
	ready,
	failed,
};
PipelineState g_pipeline = PipelineState::uninitialized;
std::condition_variable g_pipeline_cv;
std::optional<bpftime::attach::sass_aot::SassAotResult> g_compiled;

// True on a thread that is currently inside the BPF-derived SASS callback
// launched by this interposer. The callback's internal cuModuleGetFunction
// and cuLaunchKernel calls must not re-enter the interposition path.
thread_local bool g_interposing = false;

void log(const std::string &message)
{
	::fprintf(stderr, "[sass_aot_interposer] %s\n", message.c_str());
}

// Caller holds g_mutex.
void read_config_locked()
{
	if (g_config_read)
		return;
	g_config_read = true;
	const char *target = std::getenv("BPFTIME_SASS_AOT_TARGET");
	if (target == nullptr || target[0] == '\0')
		return; // transparent
	g_target = target;
	const char *object = std::getenv("BPFTIME_SASS_AOT_BPF_OBJECT");
	const char *out_dir = std::getenv("BPFTIME_SASS_AOT_OUT_DIR");
	if (object == nullptr || object[0] == '\0' || out_dir == nullptr ||
	    out_dir[0] == '\0') {
		log("BPFTIME_SASS_AOT_TARGET is set but "
		    "BPFTIME_SASS_AOT_BPF_OBJECT/BPFTIME_SASS_AOT_OUT_DIR are "
		    "missing; interposition disabled");
		g_target.clear();
		return;
	}
	g_bpf_object = object;
	g_out_dir = out_dir;
	const char *section = std::getenv("BPFTIME_SASS_AOT_SECTION");
	if (section != nullptr && section[0] != '\0')
		g_bpf_section = section;
}

// Strict GPU verifier -> ptxpass eBPF-to-PTX -> ptxas, exactly once per
// process. Returns true when the compiled result is ready. The lock is
// released while the pipeline (file I/O, fork/exec ptxas) runs; a concurrent
// caller waits on g_pipeline_cv instead of compiling again.
bool ensure_compiled()
{
	using namespace bpftime::attach::sass_aot;

	std::string bpf_object;
	std::string section;
	std::string out_dir;
	{
		std::unique_lock<std::mutex> lock(g_mutex);
		if (g_pipeline == PipelineState::uninitialized) {
			g_pipeline = PipelineState::compiling;
			bpf_object = g_bpf_object;
			section = g_bpf_section;
			out_dir = g_out_dir;
		} else {
			g_pipeline_cv.wait(lock, [] {
				return g_pipeline == PipelineState::ready ||
				       g_pipeline == PipelineState::failed;
			});
			return g_pipeline == PipelineState::ready;
		}
	}

	// Compile outside the lock.
	SassAotResult result;
	std::vector<uint64_t> words;
	std::string matched_section;
	if (const auto error =
		    load_bpf_program_words(bpf_object, section, words,
					   matched_section)) {
		result.error = *error;
	} else {
		SassAotOptions opts;
		opts.out_dir = out_dir;
		opts.func_name = "sass_aot_probe";
		opts.context_size = sizeof(uint64_t);
		result = compile_ebpf_to_sass_aot(words, matched_section, opts);
	}

	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (result.ok) {
			g_compiled = std::move(result);
			g_pipeline = PipelineState::ready;
		} else {
			g_pipeline = PipelineState::failed;
		}
		g_pipeline_cv.notify_all();
	}
	if (result.ok) {
		log("verified BPF-derived SASS callback ready: " +
		    g_compiled->cubin_path);
		return true;
	}
	log(std::string("BPF AOT compilation failed") +
	    (result.verifier_rejected ? " (verifier rejected)" : "") +
	    ": " + result.error);
	return false;
}

// Execute the verified BPF-derived SASS callback in the caller's current
// CUDA context. Runs entirely without holding g_mutex; only the snapshot of
// the compiled result is taken under the lock.
void execute_bpf_callback()
{
	using namespace bpftime::attach::sass_aot;
	if (!ensure_compiled())
		return;
	CUcontext context = nullptr;
	if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
		log("no current CUDA context at cuLaunchKernel; callback "
		    "skipped");
		return;
	}
	std::optional<SassAotResult> compiled;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		compiled = g_compiled;
	}
	std::vector<uint8_t> context_data(compiled->context_size, 0);
	const auto executed = execute_sass_aot_in_context(context, *compiled,
							  context_data);
	if (!executed.ok) {
		log("BPF-derived SASS callback failed: " + executed.error);
		return;
	}
	uint64_t value = 0;
	std::memcpy(&value, context_data.data(), sizeof(value));
	log("BPF-derived SASS callback executed (context value " +
	    std::to_string(value) + ")");
}

} // namespace

extern "C" __attribute__((visibility("default"))) CUresult
cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *sym)
{
	static RealModuleGetFunction real = nullptr;
	if (real == nullptr)
		real = reinterpret_cast<RealModuleGetFunction>(
			::dlsym(RTLD_NEXT, "cuModuleGetFunction"));
	if (real == nullptr) {
		log("cannot resolve real cuModuleGetFunction");
		return CUDA_ERROR_INVALID_CONTEXT;
	}
	const CUresult status = real(hfunc, hmod, sym);
	// g_interposing is a per-thread flag; the callback's internal
	// cuModuleGetFunction (for the BPF-derived entry) passes through
	// without bookkeeping and without taking g_mutex.
	if (status == CUDA_SUCCESS && hfunc != nullptr && sym != nullptr &&
	    !g_interposing) {
		std::lock_guard<std::mutex> lock(g_mutex);
		g_function_names[*hfunc] = sym;
	}
	return status;
}

extern "C" __attribute__((visibility("default"))) CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
	       unsigned int gridDimZ, unsigned int blockDimX,
	       unsigned int blockDimY, unsigned int blockDimZ,
	       unsigned int sharedMemBytes, CUstream hStream,
	       void **kernelParams, void **extra)
{
	static RealLaunchKernel real = nullptr;
	if (real == nullptr)
		real = reinterpret_cast<RealLaunchKernel>(
			::dlsym(RTLD_NEXT, "cuLaunchKernel"));
	if (real == nullptr) {
		log("cannot resolve real cuLaunchKernel");
		return CUDA_ERROR_INVALID_CONTEXT;
	}

	if (!g_interposing) {
		bool interpose = false;
		std::string target;
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			read_config_locked();
			if (!g_target.empty()) {
				const auto it = g_function_names.find(f);
				if (it != g_function_names.end() &&
				    it->second == g_target) {
					interpose = true;
					target = it->second;
				}
			}
		}
		if (interpose) {
			// The callback may itself call cuModuleGetFunction
			// and cuLaunchKernel (for the BPF-derived entry);
			// g_interposing keeps those internal calls out of
			// the interposition path, and execute_bpf_callback
			// never holds g_mutex across a CUDA call.
			log("launch-boundary interposition for target '" +
			    target + "'");
			g_interposing = true;
			execute_bpf_callback();
			g_interposing = false;
		}
	}
	return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
		    blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
