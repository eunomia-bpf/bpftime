#ifndef _PTX_COMPILER
#define _PTX_COMPILER

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <optional>
#include <ostream>
struct nv_attach_impl_ptx_compiler;

extern "C" {
nv_attach_impl_ptx_compiler *nv_attach_impl_create_compiler();
void nv_attach_impl_destroy_compiler(nv_attach_impl_ptx_compiler *);

int nv_attach_impl_compile(nv_attach_impl_ptx_compiler *, const char *,
			   const char **args, int arg_count);

const char *nv_attach_impl_get_error_log(nv_attach_impl_ptx_compiler *);
const char *nv_attach_impl_get_info_log(nv_attach_impl_ptx_compiler *);

int nv_attach_impl_get_compiled_program(nv_attach_impl_ptx_compiler *,
					uint8_t **, size_t *);
}

namespace bpftime
{
struct nv_attach_impl_ptx_compiler_handler {
	nv_attach_impl_ptx_compiler *(*create)();
	void (*destroy)(nv_attach_impl_ptx_compiler *);
	int (*compile)(nv_attach_impl_ptx_compiler *, const char *,
		       const char **, int);
	const char *(*get_error_log)(nv_attach_impl_ptx_compiler *);
	const char *(*get_info_log)(nv_attach_impl_ptx_compiler *);
	int (*get_compiled_program)(nv_attach_impl_ptx_compiler *, uint8_t **,
				    size_t *);
};
static inline std::optional<nv_attach_impl_ptx_compiler_handler>
load_nv_attach_impl_ptx_compiler(const char *path, void *&dl_handle)
{
	void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		std::cerr << "Unable to load dynamic library " << path << " = "
			  << dlerror() << std::endl;
		return {};
	}
	nv_attach_impl_ptx_compiler_handler result;
	result.compile = (decltype(result.compile))dlsym(
		handle, "nv_attach_impl_compile");
	result.create = (decltype(result.create))dlsym(
		handle, "nv_attach_impl_create_compiler");

	result.destroy = (decltype(result.destroy))dlsym(
		handle, "nv_attach_impl_destroy_compiler");
	result.get_error_log = (decltype(result.get_error_log))dlsym(
		handle, "nv_attach_impl_get_error_log");
	result.get_info_log = (decltype(result.get_info_log))dlsym(
		handle, "nv_attach_impl_get_info_log");
	result.get_compiled_program =
		(decltype(result.get_compiled_program))dlsym(
			handle, "nv_attach_impl_get_compiled_program");
	dl_handle = handle;
	return result;
}
} // namespace bpftime
#endif
