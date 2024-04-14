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

#ifndef LIBEBPF_H_
#define LIBEBPF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ebpf_inst.h"

/**
 * @brief Default maximum number of instructions that a program can contain.
 */
#if !defined(EBPF_MAX_INSTS)
#define EBPF_MAX_INSTS 65536
#endif

/**
 * @brief Default maximum number of helper function that a program can call.
 */
#if !defined(MAX_EXT_FUNCS)
#define MAX_EXT_FUNCS 8192
#endif

/**
 * @brief Default stack size for the VM.
 */
#if !defined(EBPF_STACK_SIZE)
#define EBPF_STACK_SIZE 512
#endif

/**
 * @brief Opaque type for the ebpf VM.
 */
struct ebpf_vm;

/**
 * @brief Opaque type for a ebpf JIT compiled function.
 */
typedef uint64_t (*ebpf_jit_fn)(void *mem, size_t mem_len);

/**
 * @brief Create a new ebpf VM.
 *
 * @return A pointer to the new VM, or NULL on failure.
 */
struct ebpf_vm *ebpf_create(void);

/**
 * @brief Free a ebpf VM.
 *
 * @param[in] vm The VM to free.
 */
void ebpf_destroy(struct ebpf_vm *vm);

/**
 * @brief Enable / disable bounds_check. Bounds check is enabled by default, but
 * it may be too restrictive.
 *
 * @param[in] vm The VM to enable / disable bounds check on.
 * @param[in] enable Enable bounds check if true, disable if false.
 * @retval true Bounds check was previously enabled.
 */
bool ebpf_toggle_bounds_check(struct ebpf_vm *vm, bool enable);

/**
 * @brief Set the function to be invoked if the program hits a fatal error.
 *
 * @param[in] vm The VM to set the error function on.
 * @param[in] error_printf The function to be invoked on fatal error.
 */
void ebpf_set_error_print(struct ebpf_vm *vm,
			  int (*error_printf)(FILE *stream, const char *format,
					      ...));

/**
 * @brief Register an external function.
 * The immediate field of a CALL instruction is an index into an array of
 * functions registered by the user. This API associates a function with
 * an index.
 *
 * @param[in] vm The VM to register the function on.
 * @param[in] index The index to register the function at.
 * @param[in] name The human readable name of the function.
 * @param[in] fn The function to register.
 * @retval 0 Success.
 * @retval -1 Failure.
 */
int ebpf_register(struct ebpf_vm *vm, unsigned int index, const char *name,
		  void *fn);

/**
 * @brief Load code into a VM.
 * This must be done before calling ebpf_exec or ebpf_compile and after
 * registering all functions.
 *
 * 'code' should point to eBPF bytecodes and 'code_len' should be the size in
 * bytes of that buffer.
 *
 * @param[in] vm The VM to load the code into.
 * @param[in] code The eBPF bytecodes to load.
 * @param[in] code_len The length of the eBPF bytecodes.
 * @param[out] errmsg The error message, if any. This should be freed by the
 * caller.
 * @retval 0 Success.
 * @retval -1 Failure.
 */
int ebpf_load(struct ebpf_vm *vm, const void *code, uint32_t code_len,
	      char **errmsg);

/**
 * @brief Unload code from a VM.
 *
 * This must be done before calling ebpf_load or ebpf_load_elf, except for the
 * first time those functions are called. It clears the VM instructions to
 * allow for new code to be loaded.
 *
 * @param[in] vm The VM to unload the code from.
 */
void ebpf_unload_code(struct ebpf_vm *vm);

/**
 * @brief Execute a BPF program in the VM using the interpreter.
 *
 * A program must be loaded into the VM and all external functions must be
 * registered before calling this function.
 *
 * @param[in] vm The VM to execute the program in.
 * @param[in] mem The memory to pass to the program.
 * @param[in] mem_len The length of the memory.
 * @param[in] bpf_return_value The value of the r0 register when the program
 * exits.
 * @retval 0 Success.
 * @retval -1 Failure.
 */
int ebpf_exec(const struct ebpf_vm *vm, void *mem, size_t mem_len,
	      uint64_t *bpf_return_value);

/**
 * @brief Compile a BPF program in the VM to native code, for jit execution.
 *
 * A program must be loaded into the VM and all external functions must be
 * registered before calling this function.
 *
 * @param[in] vm The VM to compile the program in.
 * @param[out] errmsg The error message, if any. This should be freed by the
 * caller.
 * @return ebpf_jit_fn A pointer to the compiled program, or NULL on failure.
 */
ebpf_jit_fn ebpf_compile(struct ebpf_vm *vm, char **errmsg);

/**
 * @brief Instruct the ebpf runtime to apply unwind-on-success semantics to a
 * helper function. If the function returns 0, the ebpf runtime will end
 * execution of the eBPF program and immediately return control to the caller.
 * This is used for implementing function like the "bpf_tail_call" helper.
 *
 * @param[in] vm The VM to set the unwind helper in.
 * @param[in] idx Index of the helper function to unwind on success.
 * @retval 0 Success.
 * @retval -1 Failure.
 */
int ebpf_set_unwind_function_index(struct ebpf_vm *vm, unsigned int idx);

/**
 * @brief Optional secret to improve ROP protection.
 *
 * @param[in] vm The VM to set the secret for.
 * @param[in] secret Optional secret to improve ROP protection.
 * Returns 0 on success, -1 on error (e.g. if the secret is set after
 * the instructions are loaded).
 */
int ebpf_set_pointer_secret(struct ebpf_vm *vm, uint64_t secret);

/**
 * @brief Register helper functions using the lddw instruction. See
 * https://docs.kernel.org/bpf/instruction-set.html#id15 for details.
 * All functions could be null.
 *
 * @param[in] vm The VM to set the helpers for.
 * @param[in] map_by_fd A helper to convert a 32-bit file descriptor into an
 * address of a map
 * @param[in] map_by_idx A helper to to convert a 32-bit index into an address
 * of a map
 * @param[in] map_val Helper to get the address of the first value in a given
 * map
 * @param[in] var_addr Helper to get the address of a platform variable with a
 * given id
 * @param[in] code_addr Helper to get the address of the instruction at a
 * specified relative offset in number of (64-bit) instructions
 */
void ebpf_set_lddw_helpers(struct ebpf_vm *vm, uint64_t (*map_by_fd)(uint32_t),
			   uint64_t (*map_by_idx)(uint32_t),
			   uint64_t (*map_val)(uint64_t),
			   uint64_t (*var_addr)(uint32_t),
			   uint64_t (*code_addr)(uint32_t));


/**
 * @brief Load an eBPF AOT object file into a VM.
 * The AOT object file should contain native code for the target architecture.
 * 
 * @param[in] vm The VM to load the AOT object into.
 * @param[in] buf The buffer containing the AOT object file.
 * @param[in] buf_len The length of the buffer.
 * @return ebpf_jit_fn A pointer to the compiled program, or NULL on failure.
 */
ebpf_jit_fn ebpf_load_aot_object(struct ebpf_vm *vm, const void *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif
