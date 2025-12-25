#include "nvPTXCompiler.h"
#include <clocale>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
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
		if (strcmp(args[i], "--gpu-name") == 0 && i + 1 < arg_count &&
		    args[i + 1]) {
			return std::string(args[i + 1]);
		}
	}
	return std::nullopt;
}

static std::optional<std::string> extract_gpu_name_from_ptx(const char *ptx)
{
	const char *p = ptx;
	const char *needle = ".target";
	while ((p = strstr(p, needle)) != nullptr) {
		p += strlen(needle);
		while (*p == ' ' || *p == '\t')
			p++;
		// Accept ".target sm_52" and ".target sm_52, ..."
		const char *start = p;
		while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
		       *p != ',')
			p++;
		if (p > start)
			return std::string(start, (size_t)(p - start));
	}
	return std::nullopt;
}

static std::string rewrite_ptx_target(const char *ptx,
				      const std::string &gpu_name,
				      bool &changed,
				      std::optional<std::string> &old_target)
{
	changed = false;
	old_target.reset();
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
	if (pos <= start)
		return s;
	old_target = s.substr(start, pos - start);
	if (*old_target == gpu_name)
		return s;
	s.replace(start, pos - start, gpu_name);
	changed = true;
	return s;
}

static int compile_with_ptxas(nv_attach_impl_ptx_compiler &state,
			      const char *ptx, const char **args,
			      int arg_count)
{
	state.error_log.clear();
	state.info_log.clear();
	state.compiled_program.clear();

	auto gpu_name_opt = extract_gpu_name_arg(args, arg_count);
	if (!gpu_name_opt || gpu_name_opt->empty()) {
		gpu_name_opt = extract_gpu_name_from_ptx(ptx);
	}
	if (!gpu_name_opt || gpu_name_opt->empty()) {
		state.error_log =
			"ptxas fallback requires --gpu-name=<sm_xx> option or PTX .target";
		return -1;
	}

	bool target_changed = false;
	std::optional<std::string> old_target;
	std::string ptx_to_compile =
		rewrite_ptx_target(ptx, *gpu_name_opt, target_changed, old_target);

	std::string ptxas_path = "/usr/local/cuda/bin/ptxas";
	if (access(ptxas_path.c_str(), X_OK) != 0) {
		ptxas_path = "ptxas";
	}

	char ptx_template[] = "/tmp/bpftime-ptxas-in.XXXXXX";
	int ptx_fd = mkstemp(ptx_template);
	if (ptx_fd < 0) {
		state.error_log = "mkstemp failed for PTX input";
		return -1;
	}
	std::string ptx_path = std::string(ptx_template) + ".ptx";
	rename(ptx_template, ptx_path.c_str());
	close(ptx_fd);
	{
		std::ofstream ofs(ptx_path, std::ios::binary);
		ofs << ptx_to_compile;
	}

	char out_template[] = "/tmp/bpftime-ptxas-out.XXXXXX";
	int out_fd = mkstemp(out_template);
	if (out_fd < 0) {
		unlink(ptx_path.c_str());
		state.error_log = "mkstemp failed for PTX output";
		return -1;
	}
	close(out_fd);
	std::string out_path = std::string(out_template) + ".cubin";
	rename(out_template, out_path.c_str());

	bool verbose = false;
	std::string opt_level;
	for (int i = 0; i < arg_count; i++) {
		if (!args[i])
			continue;
		if (strcmp(args[i], "--verbose") == 0)
			verbose = true;
		if (strcmp(args[i], "-O0") == 0 || strcmp(args[i], "-O1") == 0 ||
		    strcmp(args[i], "-O2") == 0 || strcmp(args[i], "-O3") == 0)
			opt_level = args[i];
	}
	if (opt_level.empty())
		opt_level = "-O3";

	std::string cmd = ptxas_path + " --gpu-name " + *gpu_name_opt + " " +
			  opt_level + " ";
	if (verbose)
		cmd += "-v ";
	cmd += "\"" + ptx_path + "\" -o \"" + out_path + "\" 2>&1";

	FILE *fp = popen(cmd.c_str(), "r");
	if (!fp) {
		unlink(ptx_path.c_str());
		unlink(out_path.c_str());
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
		state.info_log = "ptxas cmd: " + cmd + "\n" + out;
		if (target_changed && old_target) {
			state.info_log += "\nRewrote PTX .target from " +
					  *old_target + " to " + *gpu_name_opt +
					  "\n";
		}
		std::ifstream ifs(out_path, std::ios::binary);
		state.compiled_program.assign(
			std::istreambuf_iterator<char>(ifs),
			std::istreambuf_iterator<char>());
		unlink(ptx_path.c_str());
		unlink(out_path.c_str());
		if (state.compiled_program.empty()) {
			state.error_log = "ptxas produced empty output: " + cmd;
			return -1;
		}
		return 0;
	}
	if (WIFEXITED(rc) && WEXITSTATUS(rc) == 127) {
		state.error_log =
			"ptxas not found (exit 127), falling back to nvPTXCompiler. cmd: " +
			cmd + "\n" + out;
		unlink(ptx_path.c_str());
		unlink(out_path.c_str());
		return -127;
	}
	state.error_log = "ptxas failed. cmd: " + cmd + "\n" + out;
	if (target_changed && old_target) {
		state.error_log += "\nRewrote PTX .target from " + *old_target +
				   " to " + *gpu_name_opt + "\n";
	}
	unlink(ptx_path.c_str());
	unlink(out_path.c_str());
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
	if (ptr->compiler) {
		nvPTXCompilerDestroy(&ptr->compiler);
		ptr->compiler = nullptr;
	}

	// Prefer ptxas when available, as nvPTXCompiler can be unreliable for
	// some PTX variants on older toolkits (e.g. CUDA 12.6).
	{
		int ptxas_rc = compile_with_ptxas(*ptr, ptx, args, arg_count);
		if (ptxas_rc == 0) {
			return 0;
		}
		// If ptxas ran and failed, return its error output directly.
		// Falling back to nvPTXCompiler would usually call ptxas again
		// and leak errors to stderr.
		if (ptxas_rc != -127) {
			return -1;
		}
	}

	size_t len = strlen(ptx);
	if (auto err = nvPTXCompilerCreate(&ptr->compiler, len, ptx);
	    err != nvPTXCompileResult::NVPTXCOMPILE_SUCCESS) {
		ptr->error_log +=
			"Unable to create compiler: " + std::to_string((int)err) +
			"\n";
		fill_nvptx_logs(*ptr);
		if (ptr->compiler) {
			nvPTXCompilerDestroy(&ptr->compiler);
			ptr->compiler = nullptr;
		}
		return -1;
	}

	if (auto err = nvPTXCompilerCompile(ptr->compiler, arg_count, args);
	    err != NVPTXCOMPILE_SUCCESS) {
		ptr->error_log +=
			"Unable to compile: " + std::to_string((int)err) + "\n";
		fill_nvptx_logs(*ptr);
		if (ptr->compiler) {
			nvPTXCompilerDestroy(&ptr->compiler);
			ptr->compiler = nullptr;
		}
		return -1;
	}
	size_t elf_size = 0;
	nvPTXCompilerGetCompiledProgramSize(ptr->compiler, &elf_size);
	std::vector<uint8_t> compiled_program(elf_size, 0);
	nvPTXCompilerGetCompiledProgram(ptr->compiler, compiled_program.data());
	ptr->compiled_program = std::move(compiled_program);
	fill_nvptx_logs(*ptr);
	if (ptr->compiler) {
		nvPTXCompilerDestroy(&ptr->compiler);
		ptr->compiler = nullptr;
	}

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
