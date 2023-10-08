#ifndef EBPF_RUNTIME_INTERNEL_H_
#define EBPF_RUNTIME_INTERNEL_H_

#include <cstdint>
#include <cstddef>
#include "bpftime.hpp"
#include <bpftime_ffi.hpp>
namespace bpftime
{



class bpftime_prog;
class bpftime_object;

typedef bpftime_prog *bpftime_prog_ptr;

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

struct bpftime_ffi_ctx {
	struct ebpf_ffi_func_info ffi_funcs[MAX_FFI_FUNCS];
	size_t ffi_func_cnt;
};

// not used directly
#define FFI_HELPER_ID_DISPATCHER 1000
#define FFI_HELPER_ID_FIND_ID 1001

// find the ffi id from the function name
// not used directly
extern "C" uint64_t __ebpf_call_ffi_dispatcher(uint64_t id, uint64_t arg_list);

// find the ffi id from the function name
// not used directly
extern "C" int64_t __ebpf_call_find_ffi_id(const char *func_name);

extern "C" uint64_t map_ptr_by_fd(uint32_t fd);

extern "C" uint64_t map_val(uint64_t map_ptr);

} // namespace bpftime

#endif // EBPF_RUNTIME_INTERNEL_H_
