#include "nvPTXCompiler.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
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
	size_t len = strlen(ptx);
	if (auto err = nvPTXCompilerCreate(&ptr->compiler, len, ptx);
	    err != nvPTXCompileResult::NVPTXCOMPILE_SUCCESS) {
		std::cerr << "Unable to create compiler: " << (int)err
			  << std::endl;
		size_t error_size;
		if (auto err = nvPTXCompilerGetErrorLogSize(ptr->compiler,
							    &error_size);
		    err == NVPTXCOMPILE_SUCCESS) {
			if (error_size > 0) {
				char *error_log =
					(char *)malloc(error_size + 1);
				if (auto err = nvPTXCompilerGetErrorLog(
					    ptr->compiler, error_log);
				    err != NVPTXCOMPILE_SUCCESS) {
					std::cerr << "Unable to get error log"
						  << std::endl;
					free(error_log);
				} else {
					ptr->error_log = error_log;
					free(error_log);
				}
			}
		}
		return -1;
	} else {
		if (auto err = nvPTXCompilerCompile(ptr->compiler, arg_count,
						    args);
		    err != NVPTXCOMPILE_SUCCESS) {
			std::cerr << "Unable to compile: " << (int)err
				  << std::endl;
			size_t error_size = 0;
			if (nvPTXCompilerGetErrorLogSize(ptr->compiler,
							 &error_size) ==
			    NVPTXCOMPILE_SUCCESS &&
			    error_size > 0) {
				std::string log(error_size + 1, '\0');
				if (nvPTXCompilerGetErrorLog(
					    ptr->compiler, log.data()) ==
				    NVPTXCOMPILE_SUCCESS) {
					ptr->error_log = log;
				}
			}
			return -1;
		}
		size_t elf_size;
		nvPTXCompilerGetCompiledProgramSize(ptr->compiler, &elf_size);
		std::vector<uint8_t> compiled_program(elf_size, 0);
		nvPTXCompilerGetCompiledProgram(ptr->compiler,
						compiled_program.data());
		ptr->compiled_program = compiled_program;
		size_t info_size;
		nvPTXCompilerGetInfoLogSize(ptr->compiler, &info_size);
		if (info_size != 0) {
			char *info_log = (char *)malloc(info_size + 1);
			nvPTXCompilerGetInfoLog(ptr->compiler, info_log);
			ptr->info_log = info_log;
			free(info_log);
		}
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
