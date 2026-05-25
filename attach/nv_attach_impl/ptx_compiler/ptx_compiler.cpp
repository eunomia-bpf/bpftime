#include "nvPTXCompiler.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <clocale>
#include <cctype>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace {
std::string rewrite_ptx_version(std::string ptx, const std::string &version)
{
	auto pos = ptx.find(".version");
	if (pos == std::string::npos)
		return ptx;

	pos += strlen(".version");
	while (pos < ptx.size() &&
	       std::isspace(static_cast<unsigned char>(ptx[pos]))) {
		pos++;
	}

	auto start = pos;
	while (pos < ptx.size() &&
	       (std::isdigit(static_cast<unsigned char>(ptx[pos])) ||
		ptx[pos] == '.')) {
		pos++;
	}

	if (start == pos)
		return ptx;

	ptx.replace(start, pos - start, version);
	return ptx;
}

bool try_extract_supported_ptx_version(const std::string &error_log,
					       std::string &version)
{
	const char *patterns[] = { "current version is '",
				   "current version is \"" };
	for (const char *pattern : patterns) {
		auto pos = error_log.find(pattern);
		if (pos == std::string::npos)
			continue;
		pos += strlen(pattern);
		auto end = pos;
		while (end < error_log.size() &&
		       (std::isdigit(static_cast<unsigned char>(error_log[end])) ||
			error_log[end] == '.')) {
			end++;
		}
		if (end > pos) {
			version = error_log.substr(pos, end - pos);
			return true;
		}
	}
	return false;
}
} // namespace

struct nv_attach_impl_ptx_compiler {
	nvPTXCompilerHandle compiler = nullptr;

	std::string error_log;
	std::string info_log;
	std::vector<uint8_t> compiled_program;
};

extern "C" {
nv_attach_impl_ptx_compiler *nv_attach_impl_create_compiler()
{
	// nvPTXCompiler might call libc locale/ctype functions early (e.g. isalpha).
	// Initialize locale configuration before it's calling to avoid SIGSEGV in some environments.
	setlocale(LC_ALL, "");
	return new nv_attach_impl_ptx_compiler;
}

void nv_attach_impl_destroy_compiler(nv_attach_impl_ptx_compiler *ptr)
{
	if (!ptr)
		return;
	if (ptr->compiler)
		nvPTXCompilerDestroy(&ptr->compiler);
	delete ptr;
}

int nv_attach_impl_compile(nv_attach_impl_ptx_compiler *ptr, const char *ptx,
			   const char **args, int arg_count)
{
	if (!ptr || !ptx)
		return -1;

	ptr->error_log.clear();
	ptr->info_log.clear();
	ptr->compiled_program.clear();

	if (ptr->compiler)
		nvPTXCompilerDestroy(&ptr->compiler);

	std::string ptx_text = ptx;

	auto load_error_log = [&]() {
		size_t error_size = 0;
		if (ptr->compiler &&
		    nvPTXCompilerGetErrorLogSize(ptr->compiler, &error_size) ==
			    NVPTXCOMPILE_SUCCESS &&
		    error_size > 0) {
			auto *error_log = (char *)malloc(error_size + 1);
			if (error_log) {
				error_log[0] = '\0';
				if (nvPTXCompilerGetErrorLog(ptr->compiler,
							     error_log) ==
				    NVPTXCOMPILE_SUCCESS) {
					ptr->error_log = error_log;
				}
				free(error_log);
			}
		}
	};

	auto create_compiler = [&](const std::string &source) -> bool {
		if (ptr->compiler)
			nvPTXCompilerDestroy(&ptr->compiler);
		ptr->compiler = nullptr;

		if (auto err = nvPTXCompilerCreate(&ptr->compiler, source.size(),
					      source.c_str());
		    err != NVPTXCOMPILE_SUCCESS) {
			SPDLOG_ERROR("Unable to create compiler: {}",
				     static_cast<int>(err));
			load_error_log();
			return false;
		}
		return true;
	};

	auto compile_with_retry = [&]() -> bool {
		auto err = nvPTXCompilerCompile(ptr->compiler, arg_count, args);
		if (err == NVPTXCOMPILE_SUCCESS)
			return true;

		SPDLOG_ERROR("Unable to compile: {}", static_cast<int>(err));
		load_error_log();

		std::string supported_version;
		if (!try_extract_supported_ptx_version(ptr->error_log,
						      supported_version)) {
			return false;
		}

		auto rewritten_ptx =
			rewrite_ptx_version(ptx_text, supported_version);
		if (rewritten_ptx == ptx_text)
			return false;

		SPDLOG_WARN(
			"Retrying PTX compile with downgraded .version {} from compiler diagnostics",
			supported_version);
		ptx_text = std::move(rewritten_ptx);
		ptr->error_log.clear();

		if (!create_compiler(ptx_text))
			return false;

		err = nvPTXCompilerCompile(ptr->compiler, arg_count, args);
		if (err == NVPTXCOMPILE_SUCCESS)
			return true;

		SPDLOG_ERROR("Unable to compile after PTX version rewrite: {}",
			     static_cast<int>(err));
		load_error_log();
		return false;
	};

	if (!create_compiler(ptx_text)) {
		return -1;
	}

	if (!compile_with_retry()) {
		return -1;
	}

	size_t elf_size = 0;
	nvPTXCompilerGetCompiledProgramSize(ptr->compiler, &elf_size);
	std::vector<uint8_t> compiled_program(elf_size, 0);
	if (elf_size > 0) {
		nvPTXCompilerGetCompiledProgram(ptr->compiler,
						compiled_program.data());
	}
	ptr->compiled_program = std::move(compiled_program);

	size_t info_size = 0;
	nvPTXCompilerGetInfoLogSize(ptr->compiler, &info_size);
	if (info_size > 0) {
		auto *info_log = (char *)malloc(info_size + 1);
		if (info_log) {
			info_log[0] = '\0';
			if (nvPTXCompilerGetInfoLog(ptr->compiler, info_log) ==
			    NVPTXCOMPILE_SUCCESS) {
				ptr->info_log = info_log;
			}
			free(info_log);
		}
	}

	return 0;
}

const char *nv_attach_impl_get_error_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr ? ptr->error_log.c_str() : "";
}

const char *nv_attach_impl_get_info_log(nv_attach_impl_ptx_compiler *ptr)
{
	return ptr ? ptr->info_log.c_str() : "";
}

int nv_attach_impl_get_compiled_program(nv_attach_impl_ptx_compiler *ptr,
					uint8_t **dest, size_t *size)
{
	if (!ptr || !dest || !size)
		return -1;
	*dest = ptr->compiled_program.data();
	*size = ptr->compiled_program.size();
	return 0;
}
}
