#include "sass_aot.hpp"

#include <cerrno>
#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gpu_verifier.hpp"
#include "ptxpass/core.hpp"

namespace bpftime::attach::sass_aot
{
namespace
{

struct CommandResult {
	int status = -1;
	std::string output;
};

CommandResult run_captured_command(const std::vector<std::string> &args)
{
	CommandResult result;
	if (args.empty()) {
		result.output = "empty command";
		return result;
	}

	int output_pipe[2];
	if (::pipe(output_pipe) != 0) {
		result.output =
			"pipe failed: " + std::string(std::strerror(errno));
		return result;
	}

	const pid_t child = ::fork();
	if (child < 0) {
		result.output =
			"fork failed: " + std::string(std::strerror(errno));
		::close(output_pipe[0]);
		::close(output_pipe[1]);
		return result;
	}
	if (child == 0) {
		::close(output_pipe[0]);
		if (::dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
		    ::dup2(output_pipe[1], STDERR_FILENO) < 0) {
			::_exit(126);
		}
		::close(output_pipe[1]);
		std::vector<char *> argv;
		argv.reserve(args.size() + 1);
		for (const auto &arg : args)
			argv.push_back(const_cast<char *>(arg.c_str()));
		argv.push_back(nullptr);
		::execvp(argv[0], argv.data());
		::_exit(127);
	}

	::close(output_pipe[1]);
	std::ostringstream output;
	char buffer[4096];
	for (;;) {
		const ssize_t n =
			::read(output_pipe[0], buffer, sizeof(buffer));
		if (n > 0) {
			output.write(buffer, n);
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		break;
	}
	::close(output_pipe[0]);
	result.output = output.str();
	int wait_status = 0;
	pid_t waited = -1;
	do {
		waited = ::waitpid(child, &wait_status, 0);
	} while (waited < 0 && errno == EINTR);
	if (waited < 0) {
		result.status = -1;
	} else if (WIFEXITED(wait_status)) {
		result.status = WEXITSTATUS(wait_status);
	} else if (WIFSIGNALED(wait_status)) {
		result.status = 128 + WTERMSIG(wait_status);
	} else {
		result.status = -1;
	}
	return result;
}

bool write_file(const std::string &path, const std::string &content)
{
	std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
	if (!ofs)
		return false;
	ofs << content;
	return ofs.good();
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

// The generated eBPF function is emitted as a plain .func. Make it the
// cubin entry so ptxas keeps it and cuobjdump can dump its SASS. The
// function body itself is untouched compiler output.
std::string promote_func_to_entry(const std::string &ptx,
				  const std::string &func_name)
{
	const std::string func_decl = ".visible .func " + func_name + "(";
	const std::string entry_decl = ".visible .entry " + func_name + "(";
	const size_t pos = ptx.find(func_decl);
	if (pos == std::string::npos)
		return {};
	std::string entry_ptx = ptx;
	entry_ptx.replace(pos, func_decl.size(), entry_decl);
	return entry_ptx;
}

} // namespace

std::optional<std::string> load_bpf_program_words(
	const std::string &object_path, const std::string &section_name,
	std::vector<uint64_t> &words, std::string &matched_section)
{
	words.clear();
	matched_section.clear();
	if (::elf_version(EV_CURRENT) == EV_NONE)
		return "libelf initialization failed";
	const int fd = ::open(object_path.c_str(), O_RDONLY);
	if (fd < 0)
		return "cannot open BPF object: " + object_path;
	Elf *elf = elf_begin(fd, ELF_C_READ, nullptr);
	if (!elf) {
		std::string error =
			std::string("elf_begin failed: ") + elf_errmsg(-1);
		::close(fd);
		return error;
	}
	auto finish = [&](std::optional<std::string> error) {
		elf_end(elf);
		::close(fd);
		return error;
	};
	GElf_Ehdr ehdr{};
	if (elf_kind(elf) != ELF_K_ELF || gelf_getclass(elf) != ELFCLASS64 ||
	    !gelf_getehdr(elf, &ehdr) || ehdr.e_machine != EM_BPF) {
		return finish("not a 64-bit ELF BPF object: " + object_path);
	}
	size_t shstrndx = 0;
	if (elf_getshdrstrndx(elf, &shstrndx) != 0)
		return finish("cannot read ELF section-name table: " +
			      object_path);
	Elf_Scn *scn = nullptr;
	while ((scn = elf_nextscn(elf, scn)) != nullptr) {
		GElf_Shdr shdr{};
		if (!gelf_getshdr(scn, &shdr))
			continue;
		const char *name = elf_strptr(elf, shstrndx, shdr.sh_name);
		if (!name || section_name != name)
			continue;
		if (shdr.sh_type != SHT_PROGBITS ||
		    (shdr.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) !=
			    (SHF_ALLOC | SHF_EXECINSTR) ||
		    shdr.sh_size == 0 || shdr.sh_size % sizeof(uint64_t) != 0) {
			return finish("section " + std::string(name) +
				      " is not BPF instruction data");
		}
		Elf_Data *data = elf_getdata(scn, nullptr);
		if (!data || !data->d_buf || data->d_size != shdr.sh_size) {
			return finish("elf_getdata failed for section " +
				      std::string(name));
		}
		words.resize(shdr.sh_size / sizeof(uint64_t));
		std::memcpy(words.data(), data->d_buf, data->d_size);
		matched_section = name;
		return finish(std::nullopt);
	}
	return finish("no BPF program section named " + section_name + " in " +
		      object_path);
}

SassAotResult compile_ebpf_to_sass_aot(const std::vector<uint64_t> &words,
				       const std::string &section_name,
				       const SassAotOptions &opts)
{
	SassAotResult result;
	if (words.empty()) {
		result.error = "empty eBPF instruction stream";
		return result;
	}
	if (opts.out_dir.empty()) {
		result.error = "out_dir must be set";
		return result;
	}
	if (opts.sm.empty() || opts.func_name.empty()) {
		result.error = "sm and func_name must be set";
		return result;
	}

	// Stage 1: strict GPU verifier. Rejected programs stop here;
	// ptxas is never invoked.
	const auto verifier_error =
		bpftime::verifier::gpu::verify_gpu_program_with_context(
			words.data(), words.size(), section_name,
			opts.context_size);
	if (verifier_error) {
		result.verifier_rejected = true;
		result.error =
			"GPU verifier rejected program: " + *verifier_error;
		return result;
	}

	// Stage 2: eBPF -> PTX through the existing ptxpass compiler
	// (no hand-written policy PTX).
	std::string func_ptx;
	try {
		func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
			words, opts.ptx_target, opts.func_name,
			/*add_register_guard_and_filter_version_headers=*/
			false,
			/*with_arguments=*/true);
	} catch (const std::exception &ex) {
		result.error = std::string("eBPF-to-PTX compilation failed: ") +
			       ex.what();
		return result;
	} catch (...) {
		result.error = "eBPF-to-PTX compilation failed: unknown error";
		return result;
	}
	if (func_ptx.empty()) {
		result.error = "eBPF-to-PTX compilation produced no PTX";
		return result;
	}

	const std::string entry_ptx =
		promote_func_to_entry(func_ptx, opts.func_name);
	if (entry_ptx.empty()) {
		result.error = "generated PTX does not contain the function " +
			       opts.func_name + " to promote to .entry";
		return result;
	}

	// Stage 3: standalone PTX translation unit + ptxas.
	const std::string ptx = ".version " + opts.ptx_version + "\n" +
				".target " + opts.sm + "\n" +
				".address_size 64\n" + entry_ptx;
	std::error_code ec;
	std::filesystem::create_directories(opts.out_dir, ec);
	const auto ptx_path =
		std::filesystem::path(opts.out_dir) / "sass_aot.ptx";
	const auto cubin_path =
		std::filesystem::path(opts.out_dir) / "sass_aot.cubin";
	if (!write_file(ptx_path.string(), ptx)) {
		result.error = "cannot write PTX file: " + ptx_path.string();
		return result;
	}
	result.ptx_path = ptx_path.string();

	const CommandResult ptxas_result = run_captured_command(
		{ opts.ptxas_path, "-arch=" + opts.sm, ptx_path.string(), "-o",
		  cubin_path.string() });
	if (ptxas_result.status != 0 || !std::filesystem::exists(cubin_path)) {
		result.error = "ptxas failed (status " +
			       std::to_string(ptxas_result.status) +
			       "): " + ptxas_result.output;
		return result;
	}
	result.cubin_path = cubin_path.string();
	result.entry_name = opts.func_name;
	result.context_size = opts.context_size;
	result.ok = true;
	return result;
}

SassAotExecutionResult execute_sass_aot(const SassAotResult &compiled,
					std::vector<uint8_t> &context,
					const SassAotExecutionOptions &opts)
{
	SassAotExecutionResult result;
	if (!compiled.ok || compiled.cubin_path.empty() ||
	    compiled.entry_name.empty() || compiled.context_size == 0) {
		result.error = "invalid or incomplete AOT compilation result";
		return result;
	}
	if (context.size() != compiled.context_size) {
		result.error =
			"context size does not match the verified AOT ABI";
		return result;
	}
	if (opts.device_ordinal < 0) {
		result.error = "device ordinal must be non-negative";
		return result;
	}

	CUcontext cuda_context = nullptr;
	CUmodule module = nullptr;
	CUdeviceptr device_context = 0;

	auto check = [&](CUresult status, const char *operation) {
		if (status == CUDA_SUCCESS)
			return true;
		if (result.error.empty())
			result.error = cuda_error(status, operation);
		return false;
	};

	CUdevice device{};
	CUfunction entry = nullptr;
	do {
		if (!check(cuInit(0), "cuInit"))
			break;
		if (!check(cuDeviceGet(&device, opts.device_ordinal),
			   "cuDeviceGet"))
			break;
		if (!check(cuCtxCreate(&cuda_context, 0, device),
			   "cuCtxCreate"))
			break;
		if (!check(cuModuleLoad(&module, compiled.cubin_path.c_str()),
			   "cuModuleLoad"))
			break;
		if (!check(cuModuleGetFunction(&entry, module,
					       compiled.entry_name.c_str()),
			   "cuModuleGetFunction"))
			break;
		if (!check(cuMemAlloc(&device_context, context.size()),
			   "cuMemAlloc"))
			break;
		if (!check(cuMemcpyHtoD(device_context, context.data(),
					context.size()),
			   "cuMemcpyHtoD"))
			break;

		uint64_t context_size = context.size();
		void *arguments[] = { &device_context, &context_size };
		if (!check(cuLaunchKernel(entry, 1, 1, 1, 1, 1, 1, 0, nullptr,
					  arguments, nullptr),
			   "cuLaunchKernel"))
			break;
		if (!check(cuCtxSynchronize(), "cuCtxSynchronize"))
			break;
		if (!check(cuMemcpyDtoH(context.data(), device_context,
					context.size()),
			   "cuMemcpyDtoH"))
			break;
		result.ok = true;
	} while (false);

	if (device_context != 0 &&
	    !check(cuMemFree(device_context), "cuMemFree"))
		result.ok = false;
	if (module != nullptr &&
	    !check(cuModuleUnload(module), "cuModuleUnload"))
		result.ok = false;
	if (cuda_context != nullptr &&
	    !check(cuCtxDestroy(cuda_context), "cuCtxDestroy"))
		result.ok = false;
	return result;
}

SassAotExecutionResult execute_sass_aot_in_context(
    CUcontext context, const SassAotResult &compiled,
    std::vector<uint8_t> &context_data)
{
	SassAotExecutionResult result;
	if (!compiled.ok || compiled.cubin_path.empty() ||
	    compiled.entry_name.empty() || compiled.context_size == 0) {
		result.error = "invalid or incomplete AOT compilation result";
		return result;
	}
	if (context == nullptr) {
		result.error =
			"null CUDA context at the companion interposition "
			"boundary";
		return result;
	}
	if (context_data.size() != compiled.context_size) {
		result.error =
			"context size does not match the verified AOT ABI";
		return result;
	}

	CUmodule module = nullptr;
	CUdeviceptr device_context = 0;

	auto check = [&](CUresult status, const char *operation) {
		if (status == CUDA_SUCCESS)
			return true;
		if (result.error.empty())
			result.error = cuda_error(status, operation);
		return false;
	};

	CUfunction entry = nullptr;
	do {
		if (!check(cuCtxSetCurrent(context), "cuCtxSetCurrent"))
			break;
		if (!check(cuModuleLoad(&module, compiled.cubin_path.c_str()),
			   "cuModuleLoad"))
			break;
		if (!check(cuModuleGetFunction(&entry, module,
					       compiled.entry_name.c_str()),
			   "cuModuleGetFunction"))
			break;
		if (!check(cuMemAlloc(&device_context, context_data.size()),
			   "cuMemAlloc"))
			break;
		if (!check(cuMemcpyHtoD(device_context, context_data.data(),
					context_data.size()),
			   "cuMemcpyHtoD"))
			break;

		uint64_t context_size = context_data.size();
		void *arguments[] = { &device_context, &context_size };
		if (!check(cuLaunchKernel(entry, 1, 1, 1, 1, 1, 1, 0, nullptr,
					  arguments, nullptr),
			   "cuLaunchKernel"))
			break;
		// Synchronizing the caller's context also completes any
		// pending application kernels, so the boundary returns only
		// when both the application work and the BPF-derived entry
		// have finished.
		if (!check(cuCtxSynchronize(), "cuCtxSynchronize"))
			break;
		if (!check(cuMemcpyDtoH(context_data.data(), device_context,
					context_data.size()),
			   "cuMemcpyDtoH"))
			break;
		result.ok = true;
	} while (false);

	if (device_context != 0 &&
	    !check(cuMemFree(device_context), "cuMemFree"))
		result.ok = false;
	if (module != nullptr &&
	    !check(cuModuleUnload(module), "cuModuleUnload"))
		result.ok = false;
	return result;
}

std::string run_cuobjdump_sass(const std::string &cubin_path,
			       const SassAotOptions &opts)
{
	const std::string cuobjdump =
		opts.cuobjdump_path.empty() ? "cuobjdump" : opts.cuobjdump_path;
	const CommandResult result =
		run_captured_command({ cuobjdump, "-sass", cubin_path });
	return result.output;
}

std::string run_cuobjdump_symbols(const std::string &cubin_path,
				  const SassAotOptions &opts)
{
	const std::string cuobjdump =
		opts.cuobjdump_path.empty() ? "cuobjdump" : opts.cuobjdump_path;
	const CommandResult result = run_captured_command(
		{ cuobjdump, "--dump-elf-symbols", cubin_path });
	return result.output;
}

std::string run_cuobjdump_ptx(const std::string &cubin_path,
			      const SassAotOptions &opts)
{
	const std::string cuobjdump =
		opts.cuobjdump_path.empty() ? "cuobjdump" : opts.cuobjdump_path;
	const CommandResult result =
		run_captured_command({ cuobjdump, "-ptx", cubin_path });
	return result.output;
}

} // namespace bpftime::attach::sass_aot
