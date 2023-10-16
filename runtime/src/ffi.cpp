#include <cstdint>
#include <cstring>
#include "bpftime_internal.h"
#include <spdlog/spdlog.h>
namespace bpftime
{

struct bpftime_ffi_ctx global_ffi_ctx;

static struct ebpf_ffi_func_info *
ebpf_resovle_ffi_func(struct bpftime_ffi_ctx *ffi_ctx, uint64_t func_id)
{
	struct ebpf_ffi_func_info *func_list = ffi_ctx->ffi_funcs;
	const size_t N_FUNC =
		sizeof(ffi_ctx->ffi_funcs) / sizeof(struct ebpf_ffi_func_info);
	if (func_id < N_FUNC) {
		return &func_list[func_id];
	}
	return NULL;
}

// find the ffi id from the function name
extern "C" int64_t __ebpf_call_find_ffi_id(const char *func_name)
{
	struct bpftime_ffi_ctx *ffi_ctx = &global_ffi_ctx;
	assert(func_name && ffi_ctx && "Did you forget to load ebpf program?");
	struct ebpf_ffi_func_info *func_list = ffi_ctx->ffi_funcs;
	for (size_t i = 0; i < ffi_ctx->ffi_func_cnt; i++) {
		if (strcmp(func_list[i].name, func_name) == 0) {
			spdlog::info("Find func {} at {}", func_name, i);
			return i;
		}
	}
	return -1;
}

void bpftime_ffi_register_ffi(uint64_t id, ebpf_ffi_func_info func_info)
{
	global_ffi_ctx.ffi_funcs[id] = func_info;
}

int bpftime_ffi_resolve_from_info(bpf_attach_ctx *probe_ctx,
				  ebpf_ffi_func_info func_info)
{
	void *func_addr = probe_ctx->find_function_by_name(func_info.name);
	if (!func_addr) {
		spdlog::error("Failed to get function address for {}",
			      func_info.name);
		return -1;
	}
	if (global_ffi_ctx.ffi_func_cnt == MAX_FFI_FUNCS - 1) {
		spdlog::error("too many ffi funcs, {} > {}",
			      global_ffi_ctx.ffi_func_cnt, MAX_FFI_FUNCS);
		return -1;
	}
	global_ffi_ctx.ffi_funcs[global_ffi_ctx.ffi_func_cnt] = func_info;
	global_ffi_ctx.ffi_funcs[global_ffi_ctx.ffi_func_cnt].func =
		(ffi_func)func_addr;
	global_ffi_ctx.ffi_func_cnt++;
	return 0;
}

extern "C" uint64_t __ebpf_call_ffi_dispatcher(uint64_t id, uint64_t arg_list)
{
	assert(id < MAX_FFI_FUNCS);
	struct ebpf_ffi_func_info *func_info =
		ebpf_resovle_ffi_func(&global_ffi_ctx, id);
	if (!func_info || !func_info->func) {
		spdlog::error("func_info: {:x} for id {} not found",
			      (uintptr_t)func_info, id);
		return 0;
	}
	if (func_info->is_attached) {
		spdlog::error("func {} is already attached", func_info->name);
		return 0;
	}
	assert((size_t)func_info->num_args <= MAX_ARGS_COUNT);

	// Prepare arguments
	struct arg_list *raw_args = (struct arg_list *)arg_list;
	union arg_val args[MAX_ARGS_COUNT] = { { 0 } };
	union arg_val ret;
	for (int i = 0; i < func_info->num_args; i++) {
		args[i] =
			to_arg_val(func_info->arg_types[i], raw_args->args[i]);
	}

	ret.ptr = func_info->func(args[0].ptr, args[1].ptr, args[2].ptr,
				  args[3].ptr, args[4].ptr);

	// Convert the return value to the correct type
	return from_arg_val(func_info->ret_type, ret);
}

union arg_val to_arg_val(enum ffi_types type, uint64_t val)
{
	union arg_val arg {
		.uint64 = 0
	};
	switch (type) {
	case FFI_TYPE_INT8:
	case FFI_TYPE_INT16:
	case FFI_TYPE_INT32:
	case FFI_TYPE_INT64:
		arg.int64 = val;
		break;
	case FFI_TYPE_UINT64:
	case FFI_TYPE_UINT8:
	case FFI_TYPE_UINT16:
	case FFI_TYPE_UINT32:
		arg.uint64 = val;
		break;
	case FFI_TYPE_DOUBLE:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
		arg.double_val = *(double *)(uintptr_t)&val;
#pragma GCC diagnostic pop
		break;
	case FFI_TYPE_POINTER:
		arg.ptr = (void *)val;
		break;
	case FFI_TYPE_VOID:
		// No need to handle
		break;
	default:
		spdlog::error("Unknown type: {}", (int)type);
		// Handle other types
		break;
	}
	return arg;
}

uint64_t from_arg_val(enum ffi_types type, union arg_val val)
{
	switch (type) {
	case FFI_TYPE_INT8:
	case FFI_TYPE_INT16:
	case FFI_TYPE_INT32:
	case FFI_TYPE_INT64:
		return val.int64;
	case FFI_TYPE_UINT8:
	case FFI_TYPE_UINT16:
	case FFI_TYPE_UINT32:
	case FFI_TYPE_UINT64:
		return val.uint64;
	case FFI_TYPE_DOUBLE:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
		return *(uint64_t *)(uintptr_t)&val.double_val;
#pragma GCC diagnostic pop
	case FFI_TYPE_POINTER:
		return (uint64_t)val.ptr;
	case FFI_TYPE_VOID:
		// No need to handle
		break;
	default:
		spdlog::error("Unknown type: {}", (int)type);
		// Handle other types
		break;
	}
	return 0;
}

} // namespace bpftime
