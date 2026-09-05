#pragma once

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
};

struct SassAotResult {
	bool ok = false;
	// True when the strict GPU verifier rejected the program; in that
	// case ptxas is never invoked.
	bool verifier_rejected = false;
	std::string error;
	std::string ptx_path;
	std::string cubin_path;
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

// `cuobjdump -sass <cubin>` output (stdout+stderr combined).
std::string run_cuobjdump_sass(const std::string &cubin_path,
			       const SassAotOptions &opts = {});

// `cuobjdump --dump-elf-symbols <cubin>` output (stdout+stderr combined).
std::string run_cuobjdump_symbols(const std::string &cubin_path,
				  const SassAotOptions &opts = {});

} // namespace bpftime::attach::sass_aot
