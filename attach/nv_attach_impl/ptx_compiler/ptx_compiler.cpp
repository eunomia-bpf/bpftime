#include "nvPTXCompiler.h"
#include <clocale>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <optional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
struct nv_attach_impl_ptx_compiler {
	nvPTXCompilerHandle compiler = nullptr;

	std::string error_log;
	std::string info_log;
	std::vector<uint8_t> compiled_program;
};

static void fill_nvptx_logs(nv_attach_impl_ptx_compiler &state)
{
	if (!state.compiler)
		return;
	size_t error_size = 0;
	if (nvPTXCompilerGetErrorLogSize(state.compiler, &error_size) ==
		    NVPTXCOMPILE_SUCCESS &&
	    error_size > 0) {
		std::string tmp(error_size + 1, '\0');
		if (nvPTXCompilerGetErrorLog(state.compiler, tmp.data()) ==
		    NVPTXCOMPILE_SUCCESS) {
			state.error_log += tmp.c_str();
		}
	}

	size_t info_size = 0;
	if (nvPTXCompilerGetInfoLogSize(state.compiler, &info_size) ==
		    NVPTXCOMPILE_SUCCESS &&
	    info_size > 0) {
		std::string tmp(info_size + 1, '\0');
		if (nvPTXCompilerGetInfoLog(state.compiler, tmp.data()) ==
		    NVPTXCOMPILE_SUCCESS) {
			state.info_log += tmp.c_str();
		}
	}
}

static std::optional<std::string> extract_gpu_name_arg(const char **args,
						       int arg_count)
{
	for (int i = 0; i < arg_count; i++) {
		if (!args[i])
			continue;
		const char *prefix = "--gpu-name=";
		if (strncmp(args[i], prefix, strlen(prefix)) == 0) {
			return std::string(args[i] + strlen(prefix));
		}
	}
	return std::nullopt;
}

static std::string extract_opt_level(const char **args, int arg_count)
{
	for (int i = 0; i < arg_count; i++) {
		if (!args[i])
			continue;
		if (strcmp(args[i], "-O0") == 0 || strcmp(args[i], "-O1") == 0 ||
		    strcmp(args[i], "-O2") == 0 || strcmp(args[i], "-O3") == 0) {
			return std::string(args[i]);
		}
	}
	return "-O3";
}

static bool extract_verbose(const char **args, int arg_count)
{
	for (int i = 0; i < arg_count; i++) {
		if (!args[i])
			continue;
		if (strcmp(args[i], "--verbose") == 0 || strcmp(args[i], "-v") == 0)
			return true;
	}
	return false;
}

static std::string rewrite_ptx_target(const char *ptx,
				      const std::string &gpu_name)
{
	if (!ptx || gpu_name.empty())
		return {};
	std::string s(ptx);
	auto pos = s.find(".target");
	if (pos == std::string::npos)
		return s;
	pos += strlen(".target");
	while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
		pos++;
	auto start = pos;
	while (pos < s.size() && s[pos] != ' ' && s[pos] != '\t' &&
	       s[pos] != '\n' && s[pos] != '\r' && s[pos] != ',')
		pos++;
	if (pos > start) {
		s.replace(start, pos - start, gpu_name);
	}
	return s;
}

static int compile_with_ptxas(nv_attach_impl_ptx_compiler &state,
			      const char *ptx,
			      const std::string &gpu_name,
			      const std::string &opt_level,
			      bool verbose)
{
	state.error_log.clear();
	state.info_log.clear();
	state.compiled_program.clear();

	char in_template[] = "/tmp/bpftime-ptxas-in.XXXXXX";
	int in_fd = mkstemp(in_template);
	if (in_fd < 0) {
		state.error_log = "mkstemp failed for PTX input";
		return -1;
	}
	std::string in_path = std::string(in_template) + ".ptx";
	(void)rename(in_template, in_path.c_str());
	close(in_fd);
	{
		// Our passes may include a trampoline PTX with a higher .target;
		// make it match the requested gpu_name to avoid ptxas failures.
		std::string ptx_rewritten = rewrite_ptx_target(ptx, gpu_name);
		std::ofstream ofs(in_path, std::ios::binary);
		ofs << ptx_rewritten;
	}

	char out_template[] = "/tmp/bpftime-ptxas-out.XXXXXX";
	int out_fd = mkstemp(out_template);
	if (out_fd < 0) {
		(void)unlink(in_path.c_str());
		state.error_log = "mkstemp failed for PTX output";
		return -1;
	}
	std::string out_path = std::string(out_template) + ".cubin";
	(void)rename(out_template, out_path.c_str());
	close(out_fd);

	const std::string cmd =
		"ptxas --gpu-name " + gpu_name + " " + opt_level +
		(verbose ? " -v " : " ") + "\"" + in_path + "\" -o \"" + out_path +
		"\" 2>&1";

	FILE *fp = popen(cmd.c_str(), "r");
	if (!fp) {
		(void)unlink(in_path.c_str());
		(void)unlink(out_path.c_str());
		state.error_log = "popen failed for ptxas";
		return -1;
	}
	std::string out;
	char buf[4096];
	while (fgets(buf, sizeof(buf), fp) != nullptr) {
		out += buf;
	}
	int rc = pclose(fp);
	if (WIFEXITED(rc) && WEXITSTATUS(rc) == 0) {
		state.info_log = out;
		std::ifstream ifs(out_path, std::ios::binary);
		state.compiled_program.assign(
			std::istreambuf_iterator<char>(ifs),
			std::istreambuf_iterator<char>());
		(void)unlink(in_path.c_str());
		(void)unlink(out_path.c_str());
		if (state.compiled_program.empty()) {
			state.error_log = "ptxas produced empty output";
			return -1;
		}
		return 0;
	}
	state.error_log = out;
	(void)unlink(in_path.c_str());
	(void)unlink(out_path.c_str());
	if (WIFEXITED(rc) && WEXITSTATUS(rc) == 127) {
		return -127;
	}
	return -1;
}

extern "C" {
nv_attach_impl_ptx_compiler *nv_attach_impl_create_compiler()
{
	/// ncPTXCompiler might call `isalpha`, before TLS was initialized.
	/// Initialize locale configuration before it's calling to avoid SIGSEGV
	setlocale(LC_ALL, "");
	auto *result = new nv_attach_impl_ptx_compiler;
	return result;
}
void nv_attach_impl_destroy_compiler(nv_attach_impl_ptx_compiler *ptr)
{
	if (ptr->compiler) {
		nvPTXCompilerDestroy(&ptr->compiler);
	}
	delete ptr;
}

int nv_attach_impl_compile(nv_attach_impl_ptx_compiler *ptr, const char *ptx,
			   const char **args, int arg_count)
{
	ptr->error_log.clear();
	ptr->info_log.clear();
	ptr->compiled_program.clear();

	// Prefer invoking ptxas directly: nvPTXCompiler can intermittently fail
	// (e.g., return code 3) on some toolkits/GPUs for PTX generated by our passes.
	auto gpu_name = extract_gpu_name_arg(args, arg_count);
	if (gpu_name && !gpu_name->empty()) {
		const auto opt_level = extract_opt_level(args, arg_count);
		const auto verbose = extract_verbose(args, arg_count);
		int rc = compile_with_ptxas(*ptr, ptx, *gpu_name, opt_level, verbose);
		if (rc == 0) {
			return 0;
		}
		if (rc != -127) {
			return -1;
		}
	}

	size_t len = strlen(ptx);
	if (auto err = nvPTXCompilerCreate(&ptr->compiler, len, ptx);
	    err != nvPTXCompileResult::NVPTXCOMPILE_SUCCESS) {
		ptr->error_log =
			"Unable to create compiler: " + std::to_string((int)err) + "\n";
		fill_nvptx_logs(*ptr);
		return -1;
	}

	if (auto err = nvPTXCompilerCompile(ptr->compiler, arg_count, args);
	    err != NVPTXCOMPILE_SUCCESS) {
		ptr->error_log =
			"Unable to compile: " + std::to_string((int)err) + "\n";
		fill_nvptx_logs(*ptr);
		return -1;
	}
	size_t elf_size = 0;
	nvPTXCompilerGetCompiledProgramSize(ptr->compiler, &elf_size);
	std::vector<uint8_t> compiled_program(elf_size, 0);
	nvPTXCompilerGetCompiledProgram(ptr->compiler, compiled_program.data());
	ptr->compiled_program = std::move(compiled_program);
	fill_nvptx_logs(*ptr);

	return 0;
}

const char *nv_attach_impl_get_error_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr->error_log.c_str();
}
const char *nv_attach_impl_get_info_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr->info_log.c_str();
}

int nv_attach_impl_get_compiled_program(nv_attach_impl_ptx_compiler *ptr,
					uint8_t **dest, size_t *size)
{
	*dest = ptr->compiled_program.data();
	*size = ptr->compiled_program.size();
	return 0;
}
}
