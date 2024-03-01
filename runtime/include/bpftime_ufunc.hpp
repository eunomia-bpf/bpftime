#ifndef _BPFTIME_UFUNC_HPP
#define _BPFTIME_UFUNC_HPP

#include <cinttypes>
#include <cstddef>
#include "bpf_attach_ctx.hpp"

#define EXTENDED_HELPER_GET_ABS_PATH_ID 1003
#define EXTENDED_HELPER_PATH_JOIN_ID 1004

#define EXTENDED_HELPER_IOURING_INIT 1006
#define EXTENDED_HELPER_IOURING_SUBMIT_WRITE 1007
#define EXTENDED_HELPER_IOURING_SUBMIT_FSYNC 1008
#define EXTENDED_HELPER_IOURING_WAIT_AND_SEEN 1009
#define EXTENDED_HELPER_IOURING_SUBMIT 1010

namespace bpftime
{
constexpr const size_t MAX_FUNC_NAME_LEN = 64;
constexpr const size_t MAX_ARGS_COUNT = 6;
constexpr const size_t MAX_UFUNC_FUNCS = 8192 * 4;
enum ufunc_types {
	UFUNC_TYPE_UNKNOWN,
	UFUNC_TYPE_VOID,
	UFUNC_TYPE_INT8,
	UFUNC_TYPE_UINT8,
	UFUNC_TYPE_INT16,
	UFUNC_TYPE_UINT16,
	UFUNC_TYPE_INT32,
	UFUNC_TYPE_UINT32,
	UFUNC_TYPE_INT64,
	UFUNC_TYPE_UINT64,
	UFUNC_TYPE_FLOAT,
	UFUNC_TYPE_DOUBLE,
	UFUNC_TYPE_POINTER,
	UFUNC_TYPE_STRUCT,
};

typedef void *(*ufunc_func)(void *r1, void *r2, void *r3, void *r4, void *r5);

/* Useful for eliminating compiler warnings.  */
#define UFUNC_FN(f) ((ufunc_func)(void *)((void (*)(void))f))

struct ebpf_ufunc_func_info {
	char name[MAX_FUNC_NAME_LEN];
	ufunc_func func;
	enum ufunc_types ret_type;
	enum ufunc_types arg_types[MAX_ARGS_COUNT];
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

union arg_val to_arg_val(enum ufunc_types type, uint64_t val);

uint64_t from_arg_val(enum ufunc_types type, union arg_val val);

// register a ufunc for a program
void bpftime_ufunc_register_ufunc(uint64_t id, ebpf_ufunc_func_info func_info);

// register a ufunc for a program base on info.
// probe ctx will find the function address and fill in the func_info
int bpftime_ufunc_resolve_from_info(ebpf_ufunc_func_info func_info,
				    void *(*function_resolver)(const char *));

} // namespace bpftime
#endif
