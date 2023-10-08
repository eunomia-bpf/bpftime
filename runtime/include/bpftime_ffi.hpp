#ifndef _BPFTIME_FFI_HPP
#define _BPFTIME_FFI_HPP
#include <cinttypes>
#include <cstddef>
#include "bpf_attach_ctx.hpp"
namespace bpftime
{
constexpr const size_t MAX_FUNC_NAME_LEN = 64;
constexpr const size_t MAX_ARGS_COUNT = 6;
constexpr const size_t MAX_FFI_FUNCS = 8192 * 4;
enum ffi_types {
	FFI_TYPE_UNKNOWN,
	FFI_TYPE_VOID,
	FFI_TYPE_INT8,
	FFI_TYPE_UINT8,
	FFI_TYPE_INT16,
	FFI_TYPE_UINT16,
	FFI_TYPE_INT32,
	FFI_TYPE_UINT32,
	FFI_TYPE_INT64,
	FFI_TYPE_UINT64,
	FFI_TYPE_FLOAT,
	FFI_TYPE_DOUBLE,
	FFI_TYPE_POINTER,
	FFI_TYPE_STRUCT,
};

typedef void *(*ffi_func)(void *r1, void *r2, void *r3, void *r4, void *r5);

/* Useful for eliminating compiler warnings.  */
#define FFI_FN(f) ((ffi_func)(void *)((void (*)(void))f))

struct ebpf_ffi_func_info {
	char name[MAX_FUNC_NAME_LEN];
	ffi_func func;
	enum ffi_types ret_type;
	enum ffi_types arg_types[MAX_ARGS_COUNT];
	int num_args;
	int id;
	bool is_attached;
};

struct arg_list {
	uint64_t args[MAX_ARGS_COUNT];
};

union arg_val {
	uint64_t uint64;
	int64_t int64;
	double double_val;
	void *ptr;
};

union arg_val to_arg_val(enum ffi_types type, uint64_t val);

uint64_t from_arg_val(enum ffi_types type, union arg_val val);

// register a ffi for a program
void bpftime_ffi_register_ffi(uint64_t id, ebpf_ffi_func_info func_info);

// register a ffi for a program base on info.
// probe ctx will find the function address and fill in the func_info
int bpftime_ffi_resolve_from_info(bpf_attach_ctx *probe_ctx,
				  ebpf_ffi_func_info func_info);

} // namespace bpftime
#endif
