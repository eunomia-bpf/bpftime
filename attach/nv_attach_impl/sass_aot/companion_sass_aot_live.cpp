// Companion/interposition live acceptance executable.
//
// Demonstrates the documented host-side module interposition boundary for
// BPF-derived SASS:
//
//   1. The host application owns its CUDA context and loads its own
//      PTX-free (SASS-only) cubin module - the "existing application".
//   2. The application launches its own kernel (original execution, in
//      flight on the context's default stream).
//   3. At the documented interposition boundary, the verified BPF-derived
//      cubin is loaded into the *same* context as a companion module and its
//      entry is invoked. No application binary, module, or SASS is rewritten.
//   4. Both results are read back: the application's own result must be
//      preserved (7) and the BPF-derived entry must have run (42).
//
// Usage: bpftime_sass_aot_companion_live [artifact-directory]
//                                             [device-ordinal]
#include "sass_aot.hpp"
#include "sass_aot_test_config.hpp"

#include <cuda.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using namespace bpftime::attach::sass_aot;

[[noreturn]] void fail(const std::string &message)
{
	std::cerr << message << '\n';
	std::exit(1);
}

std::string cuda_error(CUresult error, const char *operation)
{
	const char *name = nullptr;
	const char *description = nullptr;
	(void)cuGetErrorName(error, &name);
	(void)cuGetErrorString(error, &description);
	return std::string(operation) +
	       " failed: " + (name != nullptr ? name : "CUDA_ERROR_UNKNOWN") +
	       " (" +
	       (description != nullptr ? description : "no description") + ")";
}

} // namespace

int main(int argc, char **argv)
{
	if (argc > 3) {
		std::cerr << "usage: " << argv[0]
			  << " [artifact-directory] [device-ordinal]\n";
		return 2;
	}

	int device_ordinal = 0;
	if (argc >= 3) {
		try {
			device_ordinal = std::stoi(argv[2]);
		} catch (...) {
			std::cerr << "device ordinal must be an integer\n";
			return 2;
		}
	}

	// Phase 0: verified AOT compilation of the BPF program.
	// strict GPU verifier -> ptxpass eBPF-to-PTX -> ptxas sm_120 cubin.
	// A verifier-rejected program stops here; it never reaches the
	// interposition boundary.
	const std::string out_dir =
		argc >= 2 ? argv[1] : "/tmp/bpftime_sass_aot_companion";
	SassAotOptions compile_options;
	compile_options.out_dir = out_dir;
	compile_options.func_name = "sass_aot_probe";
	compile_options.context_size = sizeof(uint64_t);

	std::vector<uint64_t> words;
	std::string section;
	if (const auto error =
		    load_bpf_program_words(SASS_AOT_SPIKE_BPF_OBJECT,
					   "cuda__/sass_aot", words, section)) {
		fail(*error);
	}
	const auto compiled =
		compile_ebpf_to_sass_aot(words, section, compile_options);
	if (!compiled.ok) {
		fail(std::string("BPF AOT compilation failed") +
		       (compiled.verifier_rejected ? " (verifier rejected)" : "") +
		       ": " + compiled.error);
	}

	// Application CUDA context: owned by the host application.
	CUcontext context = nullptr;
	do {
		if (const auto err = cuInit(0); err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuInit"));
		CUdevice device{};
		if (const auto err = cuDeviceGet(&device, device_ordinal);
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuDeviceGet"));
		if (const auto err = cuCtxCreate(&context, 0, device);
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuCtxCreate"));

		// Original application execution: load the application's own
		// PTX-free (SASS-only) module.
		CUmodule app_module = nullptr;
		if (const auto err =
				cuModuleLoad(&app_module,
					     SASS_AOT_COMPANION_APP_CUBIN);
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuModuleLoad (application)"));

		// Negative check: the BPF-derived entry must NOT be part of
		// the application's own module. This is runtime evidence the
		// application module was not rewritten or injected; the BPF
		// entry lives only in the companion module loaded later.
		CUfunction app_probe = nullptr;
		const CUresult probe_error =
			cuModuleGetFunction(&app_probe, app_module,
					    "sass_aot_probe");
		if (probe_error != CUDA_ERROR_NOT_FOUND)
			fail("application module unexpectedly contains the "
			     "BPF-derived entry (SASS rewriting suspected)");

		CUfunction app_entry = nullptr;
		if (const auto err =
				cuModuleGetFunction(&app_entry, app_module,
						    "companion_app_kernel");
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuModuleGetFunction (app)"));

		CUdeviceptr app_out = 0;
		if (const auto err = cuMemAlloc(&app_out, sizeof(uint64_t));
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuMemAlloc (app)"));
		void *app_args[] = { &app_out };
		if (const auto err = cuLaunchKernel(
				app_entry, 1, 1, 1, 1, 1, 1, 0, nullptr,
				app_args, nullptr);
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuLaunchKernel (app)"));

		// The application kernel is now in flight on the context's
		// default stream. The application has not synchronized yet.

		// ------------------------------------------------------------------
		// Documented interposition boundary.
		//
		// A production deployment would reach this point through a
		// CUDA module-load hook (driver-API interposition, a fatbin
		// registrar, or the attach framework's module load flow). The
		// boundary function is
		// sass_aot::execute_sass_aot_in_context: it loads the
		// verified BPF-derived SASS cubin into the application's
		// active context as a companion module, invokes the BPF
		// entry, and returns after synchronizing the context.
		// ------------------------------------------------------------------
		std::vector<uint8_t> bpf_context(compiled.context_size, 0);
		const auto bpf_executed =
			execute_sass_aot_in_context(context, compiled,
						    bpf_context);
		if (!bpf_executed.ok)
			fail(std::string("interposition boundary failed: ") +
			       bpf_executed.error);
		// ------------------------------------------------------------------

		// Application checkpoint: by now the boundary has completed
		// both the application kernel and the BPF-derived entry.
		if (const auto err = cuCtxSynchronize(); err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuCtxSynchronize"));

		uint64_t app_result = 0;
		if (const auto err =
				cuMemcpyDtoH(&app_result, app_out,
					      sizeof(app_result));
		    err != CUDA_SUCCESS)
			fail(cuda_error(err, "cuMemcpyDtoH (app)"));
		if (app_result != 7)
			fail("original application execution was not "
			     "preserved: got " +
			     std::to_string(app_result) + " (expected 7)");

		uint64_t bpf_result = 0;
		std::memcpy(&bpf_result, bpf_context.data(),
			    sizeof(bpf_result));
		if (bpf_result != 42)
			fail("BPF-derived companion entry did not run: got " +
			     std::to_string(bpf_result) + " (expected 42)");

		std::cout << "companion application result: " << app_result
			  << " (original execution preserved)\n";
		std::cout << "bpf-derived companion entry result: "
			  << bpf_result << '\n';
		std::cout << "scope: companion SASS module in the "
				     "application CUDA context; no application "
				     "SASS rewriting\n";

		(void)cuMemFree(app_out);
		(void)cuModuleUnload(app_module);
	} while (false);
	(void)cuCtxDestroy(context);
	return 0;
}
