/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_ufunc.hpp"
#include <cstdint>
#include <cstring>
#include "bpftime_internal.h"
#include <spdlog/spdlog.h>

namespace bpftime
{

struct bpftime_ufunc_ctx global_ufunc_ctx;

static struct ebpf_ufunc_func_info *
ebpf_resovle_ufunc_func(struct bpftime_ufunc_ctx *ufunc_ctx, uint64_t func_id)
{
	struct ebpf_ufunc_func_info *func_list = ufunc_ctx->ufunc_funcs;
	const size_t N_FUNC = sizeof(ufunc_ctx->ufunc_funcs) /
			      sizeof(struct ebpf_ufunc_func_info);
	if (func_id < N_FUNC) {
		return &func_list[func_id];
	}
	return NULL;
}

// find the ufunc id from the function name
extern "C" int64_t __ebpf_call_find_ufunc_id(const char *func_name)
{
	struct bpftime_ufunc_ctx *ufunc_ctx = &global_ufunc_ctx;
	if (!func_name || !ufunc_ctx) {
		SPDLOG_ERROR("Invalid func_name or ufunc_ctx");
		return -1;
	}
	struct ebpf_ufunc_func_info *func_list = ufunc_ctx->ufunc_funcs;
	for (size_t i = 0; i < ufunc_ctx->ufunc_func_cnt; i++) {
		if (strcmp(func_list[i].name, func_name) == 0) {
			SPDLOG_INFO("Find func {} at {}", func_name, i);
			return i;
		}
	}
	return -1;
}

void bpftime_ufunc_register_ufunc(uint64_t id, ebpf_ufunc_func_info func_info)
{
	global_ufunc_ctx.ufunc_funcs[id] = func_info;
}

int bpftime_ufunc_resolve_from_info(ebpf_ufunc_func_info func_info,
				    void *(*function_resolver)(const char *))
{
	void *func_addr = function_resolver(func_info.name);
	if (!func_addr) {
		SPDLOG_ERROR("Failed to get function address for {}",
			     func_info.name);
		return -1;
	}
	if (global_ufunc_ctx.ufunc_func_cnt == MAX_UFUNC_FUNCS - 1) {
		SPDLOG_ERROR("too many ufunc funcs, {} > {}",
			     global_ufunc_ctx.ufunc_func_cnt, MAX_UFUNC_FUNCS);
		return -1;
	}
	global_ufunc_ctx.ufunc_funcs[global_ufunc_ctx.ufunc_func_cnt] =
		func_info;
	global_ufunc_ctx.ufunc_funcs[global_ufunc_ctx.ufunc_func_cnt].func =
		(ufunc_func)func_addr;
	global_ufunc_ctx.ufunc_func_cnt++;
	return 0;
}

extern "C" uint64_t __ebpf_call_ufunc_dispatcher(uint64_t id, uint64_t arg_list)
{
	if (id >= MAX_UFUNC_FUNCS) {
		SPDLOG_ERROR("ufunc id {} is too large, max {}", id,
			     MAX_UFUNC_FUNCS);
		return -1;
	}
	struct ebpf_ufunc_func_info *func_info =
		ebpf_resovle_ufunc_func(&global_ufunc_ctx, id);
	if (!func_info || !func_info->func) {
		SPDLOG_ERROR("func_info: {:x} for id {} not found",
			     (uintptr_t)func_info, id);
		return 0;
	}
	if (func_info->is_attached) {
		SPDLOG_ERROR("func {} is already attached", func_info->name);
		return 0;
	}
	if ((size_t)func_info->num_args > MAX_ARGS_COUNT) {
		SPDLOG_ERROR("Too many arguments for func id {}", id);
		return (uint64_t)-1;
	}

	// Prepare arguments
	struct arg_list *raw_args = (struct arg_list *)arg_list;
	union arg_val args[MAX_ARGS_COUNT] = { { 0 } };
	union arg_val ret;
	for (int i = 0; i < func_info->num_args; i++) {
		args[i] =
			to_arg_val(func_info->arg_types[i], raw_args->args[i]);
	}
	SPDLOG_DEBUG("Call ufunc {}, args: {:x}, {:x}, {:x}, {:x}, {:x}", id,
		     args[0].uint64, args[1].uint64, args[2].uint64,
		     args[3].uint64, args[4].uint64);
	ret.ptr = func_info->func(args[0].ptr, args[1].ptr, args[2].ptr,
				  args[3].ptr, args[4].ptr);

	// Convert the return value to the correct type
	return from_arg_val(func_info->ret_type, ret);
}

union arg_val to_arg_val(enum ufunc_types type, uint64_t val)
{
	union arg_val arg {
		.uint64 = 0
	};
	switch (type) {
	case UFUNC_TYPE_INT8:
	case UFUNC_TYPE_INT16:
	case UFUNC_TYPE_INT32:
	case UFUNC_TYPE_INT64:
		arg.int64 = val;
		break;
	case UFUNC_TYPE_UINT64:
	case UFUNC_TYPE_UINT8:
	case UFUNC_TYPE_UINT16:
	case UFUNC_TYPE_UINT32:
		arg.uint64 = val;
		break;
	case UFUNC_TYPE_DOUBLE:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
		arg.double_val = *(double *)(uintptr_t)&val;
#pragma GCC diagnostic pop
		break;
	case UFUNC_TYPE_POINTER:
		arg.ptr = (void *)val;
		break;
	case UFUNC_TYPE_VOID:
		// No need to handle
		break;
	default:
		SPDLOG_ERROR("Unknown type: {}", (int)type);
		// Handle other types
		break;
	}
	return arg;
}

uint64_t from_arg_val(enum ufunc_types type, union arg_val val)
{
	switch (type) {
	case UFUNC_TYPE_INT8:
	case UFUNC_TYPE_INT16:
	case UFUNC_TYPE_INT32:
	case UFUNC_TYPE_INT64:
		return val.int64;
	case UFUNC_TYPE_UINT8:
	case UFUNC_TYPE_UINT16:
	case UFUNC_TYPE_UINT32:
	case UFUNC_TYPE_UINT64:
		return val.uint64;
	case UFUNC_TYPE_DOUBLE:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
		return *(uint64_t *)(uintptr_t)&val.double_val;
#pragma GCC diagnostic pop
	case UFUNC_TYPE_POINTER:
		return (uint64_t)val.ptr;
	case UFUNC_TYPE_VOID:
		// No need to handle
		break;
	default:
		SPDLOG_ERROR("Unknown type: {}", (int)type);
		// Handle other types
		break;
	}
	return 0;
}

} // namespace bpftime
