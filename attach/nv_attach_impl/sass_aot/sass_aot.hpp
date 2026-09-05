#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bpftime::attach::sass_aot
{

struct SassAotOptions {
	// ptxas -arch target; generates the cubin SASS artifact
	std::string sm = "sm_120";
	// PTX ISA version written to the generated translation unit header.
	// sm_120 requires PTX ISA 8.7 (CUDA 12.9).
	std::string ptx_version = "8.7";
	// NVPTX CPU handed to the LLVM PTX backend when translating eBPF to
	// PTX. PTX is forward-compatible, so this may target an older SM
	// than `sm` (which ptxas compiles to SASS). Defaults to sm_90,
	// supported by LLVM 17+.
	std::string ptx_target = "sm_90";
	std::string func_name = "sass_aot_probe";
	// Writable directory receiving <out_dir>/sass_aot.ptx and
	// <out_dir>/sass_aot.cubin
	std::string out_dir;
	std::string ptxas_path = "ptxas";
	std::string cuobjdump_path = "cuobjdump";
	// Size of the read/write context exposed to the eBPF program as R1.
	// The strict verifier proves accesses against this exact bound.
	size_t context_size = sizeof(uint64_t);
};

struct SassAotResult {
	bool ok = false;
	// True when the strict GPU verifier rejected the program; in that
	// case ptxas is never invoked.
	bool verifier_rejected = false;
	std::string error;
	std::string ptx_path;
	std::string cubin_path;
	std::string entry_name;
	size_t context_size = 0;
};

struct SassAotExecutionOptions {
	int device_ordinal = 0;
};

struct SassAotExecutionResult {
	bool ok = false;
	std::string error;
};

// Extract the eBPF instruction words of the program whose section name is
// `section_name` from a real BPF ELF object file (e.g. clang -target bpf
// output). Returns an error string, or std::nullopt on success.
std::optional<std::string> load_bpf_program_words(
	const std::string &object_path, const std::string &section_name,
	std::vector<uint64_t> &words, std::string &matched_section);

// Strict GPU verifier -> existing ptxpass eBPF-to-PTX compiler ->
// ptxas -arch=<sm> -> cubin. ptxas is only invoked for verifier-accepted
// programs.
SassAotResult compile_ebpf_to_sass_aot(const std::vector<uint64_t> &words,
				       const std::string &section_name,
				       const SassAotOptions &opts = {});

// Load a verified AOT cubin with the CUDA Driver API, launch its generated
// eBPF entry once, synchronize, and copy the context back to the host. This is
// a standalone execution path; it does not inject the entry into an existing
// application's fatbin.
SassAotExecutionResult
execute_sass_aot(const SassAotResult &compiled, std::vector<uint8_t> &context,
		 const SassAotExecutionOptions &opts = {});

// Documented host-side module interposition boundary.
//
// `context` is the CUDA context owned by a host application whose own
// (PTX-free, SASS) module(s) may already be loaded and whose own kernels may
// already be in flight. The verified BPF-derived SASS cubin is loaded into
// that same context as a *companion* module; its entry is launched on the
// context's default stream with `context_data` as the verified context, the
// context is synchronized, and `context_data` is read back.
//
// This is a companion/interposed module path, not a binary-rewriting path:
// the application's own modules, cubin, and SASS are never modified, and the
// BPF-derived code runs only through this boundary in a sibling module.
// Because the boundary synchronizes the caller's context, any pending
// application work also completes when it returns.
//
// Fail-closed: all inputs are validated before any CUDA call. A program the
// strict verifier rejected (or any incomplete compilation result) is never
// loaded.
SassAotExecutionResult execute_sass_aot_in_context(
    CUcontext context, const SassAotResult &compiled,
    std::vector<uint8_t> &context_data);

// `cuobjdump -sass <cubin>` output (stdout+stderr combined).
std::string run_cuobjdump_sass(const std::string &cubin_path,
			       const SassAotOptions &opts = {});

// `cuobjdump --dump-elf-symbols <cubin>` output (stdout+stderr combined).
std::string run_cuobjdump_symbols(const std::string &cubin_path,
				  const SassAotOptions &opts = {});

// `cuobjdump -ptx <cubin>` output (stdout+stderr combined). For a ptxas-built
// SASS-only cubin this is empty, which is how tests prove an application
// artifact is PTX-free.
std::string run_cuobjdump_ptx(const std::string &cubin_path,
			      const SassAotOptions &opts = {});

} // namespace bpftime::attach::sass_aot
