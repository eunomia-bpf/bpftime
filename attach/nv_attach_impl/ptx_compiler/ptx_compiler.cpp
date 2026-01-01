#include "nvPTXCompiler.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <clocale>
#include <iostream>
#include <string>
#include <vector>

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

	const size_t len = strlen(ptx);
	if (auto err = nvPTXCompilerCreate(&ptr->compiler, len, ptx);
	    err != NVPTXCOMPILE_SUCCESS) {
		std::cerr << "Unable to create compiler: " << (int)err
			  << std::endl;
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
		return -1;
	}

	if (auto err = nvPTXCompilerCompile(ptr->compiler, arg_count, args);
	    err != NVPTXCOMPILE_SUCCESS) {
		std::cerr << "Unable to compile: " << (int)err << std::endl;
		size_t error_size = 0;
		if (nvPTXCompilerGetErrorLogSize(ptr->compiler, &error_size) ==
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
