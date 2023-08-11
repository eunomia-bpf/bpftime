#include "ebpf_inst.h"
#include "llvm_bpf_jit.h"
#include "llvm_jit_context.h"
#include "bpf_jit_helpers.h"
#include <iterator>

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Error.h>
#include <llvm/ExecutionEngine/JITSymbol.h>

#include <utility>
#include <iostream>
#include <string>
#include <cinttypes>
#include <cstdarg>
using namespace llvm;
using namespace llvm::orc;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copyright 2015 Big Switch Networks, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
void ebpf_store_instruction(const struct ebpf_vm *vm, uint16_t pc,
			    struct ebpf_inst inst);

char *ebpf_error(const char *fmt, ...)
{
	char *msg;
	va_list ap;
	va_start(ap, fmt);
	if (vasprintf(&msg, fmt, ap) < 0) {
		msg = NULL;
	}
	va_end(ap);
	return msg;
}

bool ebpf_toggle_bounds_check(struct ebpf_vm *vm, bool enable)
{
	bool old = vm->bounds_check_enabled;
	vm->bounds_check_enabled = enable;
	return old;
}

void ebpf_set_error_print(struct ebpf_vm *vm,
			  int (*error_printf)(FILE *stream, const char *format,
					      ...))
{
	if (error_printf)
		vm->error_printf = error_printf;
	else
		vm->error_printf = fprintf;
}

struct ebpf_vm *ebpf_create(void)
{
	struct ebpf_vm *vm =
		static_cast<struct ebpf_vm *>(calloc(1, sizeof(*vm)));
	if (vm == NULL) {
		return NULL;
	}

	vm->ext_func_names = static_cast<const char **>(
		calloc(MAX_EXT_FUNCS, sizeof(*vm->ext_func_names)));
	if (vm->ext_func_names == NULL) {
		ebpf_destroy(vm);
		return NULL;
	}

	vm->jit_context = new bpf_jit_context(vm);

	vm->bounds_check_enabled = true;
	vm->unwind_stack_extension_index = -1;
	return vm;
}

void ebpf_destroy(struct ebpf_vm *vm)
{
	ebpf_unload_code(vm);
	free(vm->ext_func_names);
	delete vm->jit_context;
	free(vm);
}

int ebpf_register(struct ebpf_vm *vm, unsigned int idx, const char *name,
		  void *fn)
{
	if (idx >= MAX_EXT_FUNCS) {
		return -1;
	}

	vm->ext_funcs[idx] = (ext_func)fn;
	vm->ext_func_names[idx] = name;
	LOG_DEBUG("ebpf_register: %s idx: %d func: %ld\n", name, idx, (long)fn);
	return 0;
}

int ebpf_set_unwind_function_index(struct ebpf_vm *vm, unsigned int idx)
{
	if (vm->unwind_stack_extension_index != -1) {
		return -1;
	}

	vm->unwind_stack_extension_index = idx;
	return 0;
}

unsigned int ebpf_lookup_registered_function(struct ebpf_vm *vm,
					     const char *name)
{
	int i;
	for (i = 0; i < MAX_EXT_FUNCS; i++) {
		const char *other = vm->ext_func_names[i];
		if (other && !strcmp(other, name)) {
			return i;
		}
	}
	return -1;
}

int ebpf_load(struct ebpf_vm *vm, const void *code, uint32_t code_len,
	      char **errmsg)
{
	const struct ebpf_inst *source_inst = (const struct ebpf_inst *)code;
	*errmsg = NULL;

	if (vm->insnsi) {
		*errmsg = ebpf_error(
			"code has already been loaded into this VM. Use ebpf_unload_code() if you need to reuse this VM");
		return -1;
	}

	if (code_len % 8 != 0) {
		*errmsg = ebpf_error("code_len must be a multiple of 8");
		return -1;
	}
	vm->insnsi = (ebpf_inst *)malloc(code_len);
	if (vm->insnsi == NULL) {
		*errmsg = ebpf_error("out of memory");
		return -1;
	}

	vm->num_insts = code_len / sizeof(vm->insnsi[0]);
	// Store instructions in the vm.
	for (uint32_t i = 0; i < vm->num_insts; i++) {
		ebpf_store_instruction(vm, i, source_inst[i]);
	}

	return 0;
}

void ebpf_unload_code(struct ebpf_vm *vm)
{
	if (vm->jitted_function) {
		vm->jitted_function = NULL;
	}
	if (vm->insnsi) {
		free(vm->insnsi);
		vm->insnsi = NULL;
		vm->num_insts = 0;
	}
}

#define IS_ALIGNED(x, a) (((uintptr_t)(x) & ((a)-1)) == 0)

#if DEBUG
void ebpf_set_registers(struct ebpf_vm *vm, uint64_t *regs)
{
	vm->regs = regs;
}

uint64_t *ebpf_get_registers(const struct ebpf_vm *vm)
{
	return vm->regs;
}
#else
void ebpf_set_registers(struct ebpf_vm *vm, uint64_t *regs)
{
	(void)vm;
	(void)regs;
	fprintf(stderr,
		"ebpf warning: registers are not exposed in release mode. Please recompile in debug mode\n");
}

uint64_t *ebpf_get_registers(const struct ebpf_vm *vm)
{
	(void)vm;
	fprintf(stderr,
		"ebpf warning: registers are not exposed in release mode. Please recompile in debug mode\n");
	return NULL;
}

#endif

typedef struct _ebpf_encoded_inst {
	union {
		uint64_t value;
		struct ebpf_inst inst;
	};
} ebpf_encoded_inst;

struct ebpf_inst ebpf_fetch_instruction(const struct ebpf_vm *vm, uint16_t pc)
{
	// XOR instruction with base address of vm.
	// This makes ROP attack more difficult.
	ebpf_encoded_inst encode_inst;
	encode_inst.inst = vm->insnsi[pc];
	// encode_inst.value ^= (uint64_t)vm->insnsi;
	// encode_inst.value ^= vm->pointer_secret;
	return encode_inst.inst;
}

void ebpf_store_instruction(const struct ebpf_vm *vm, uint16_t pc,
			    struct ebpf_inst inst)
{
	// XOR instruction with base address of vm.
	// This makes ROP attack more difficult.
	ebpf_encoded_inst encode_inst;
	encode_inst.inst = inst;
	// encode_inst.value ^= (uint64_t)vm->insnsi;
	// encode_inst.value ^= vm->pointer_secret;
	vm->insnsi[pc] = encode_inst.inst;
}

int ebpf_set_pointer_secret(struct ebpf_vm *vm, uint64_t secret)
{
	if (vm->insnsi) {
		return -1;
	}
	vm->pointer_secret = secret;
	return 0;
}

ebpf_jit_fn ebpf_compile(struct ebpf_vm *vm, char **errmsg)
{
	auto func = vm->jit_context->compile();
	vm->jitted_function = func;
	return func;
}

int ebpf_exec(const struct ebpf_vm *vm, void *mem, size_t mem_len,
	      uint64_t *bpf_return_value)
{
	bpf_jit_context *jit_context;
	if (vm->jitted_function) {
		// has jit yet
		auto ret = vm->jitted_function(mem,
					       static_cast<uint64_t>(mem_len));
		*bpf_return_value = ret;
		return 0;
	}
    // compile and run
	auto jit_vm = const_cast<struct ebpf_vm *>(vm);
    auto func = ebpf_compile(jit_vm, nullptr);
    if (!func) {
        return -1;
    }
    // after compile, run
	return ebpf_exec(vm, mem, mem_len, bpf_return_value);
}

/* For testing, this changes the mapping between x86 and eBPF registers */
void ebpf_set_register_offset(int x)
{
	// DO NOTHING because llvm handles the map
}

#ifdef __cplusplus
}
#endif