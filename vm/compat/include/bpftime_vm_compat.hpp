#ifndef _BPFTIME_VM_COMPAT_HPP
#define _BPFTIME_VM_COMPAT_HPP
#include "spdlog/spdlog.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#ifndef EBPF_STACK_SIZE
// Compatible to C headers
#define EBPF_STACK_SIZE 512
#endif

#ifndef MAX_EXT_FUNCS
#define MAX_EXT_FUNCS 8192
#endif

namespace bpftime::vm::compat
{

using precompiled_ebpf_function = uint64_t (*)(void *mem, size_t mem_len);

class bpftime_vm_impl {
    public:
	virtual ~bpftime_vm_impl()
	{
	}
	/**
	 * @brief Get the error message object
	 *
	 * @return std::string
	 */
	virtual std::string get_error_message()
	{
		SPDLOG_CRITICAL("Not implemented yet: get_error_message");
		return "";
	}
	/**
	 * @brief Toggle whether to enable bounds_check
	 *
	 * @param enable Whether to enable
	 */
	virtual bool toggle_bounds_check(bool enable)
	{
		SPDLOG_DEBUG("Not implemented yet: toggle_bounds_check");
		return false;
	}

	/**
	 * @brief Register a C-style print function for printing error strings
	 *
	 * @param fn The function
	 */
	virtual void register_error_print_callback(int (*fn)(FILE *,
							     const char *, ...))
	{
		SPDLOG_WARN(
			"Not implemented yet: register_error_print_callback");
	}

	/**
	 * @brief Register a helper function
	 *
	 * @param index Function index
	 * @param name Name of the function
	 * @param fn Pointer of the function
	 * @return int non-zero for failure, refer to get_error_message for
	 * details
	 */
	virtual int register_external_function(size_t index,
					       const std::string &name,
					       void *fn)
	{
		SPDLOG_CRITICAL(
			"Not implemented yet: register_external_function");
		return -1;
	}

	/**
	 * @brief Load code into the vm
	 *
	 * @param code Buffer to the code
	 * @param code_len Length of the code, in bytes
	 * @return int
	 */
	virtual int load_code(const void *code, size_t code_len) = 0;

	/**
	 * @brief Unload the code
	 *
	 */
	virtual void unload_code()
	{
		SPDLOG_CRITICAL("Not implemented yet: unload_code");
	}

	/**
	 * @brief Try to execute eBPF program in intepreter mode. If not
	 * supported, use JIT
	 *
	 * @param mem Buffer to the memory of eBPF program
	 * @param mem_len Length of the memory
	 * @param bpf_return_value Return value of the eBPF program
	 * @return int 0 for success, otherwise failed.
	 */
	virtual int exec(void *mem, size_t mem_len,
			 uint64_t &bpf_return_value) = 0;

	/**
	 * @brief Compile the eBPF program, return the compiled program
	 *
	 * @return Empty optional marks a failure
	 */
	virtual std::optional<precompiled_ebpf_function> compile()
	{
		SPDLOG_CRITICAL("Not implemented yet: compile");
		return {};
	}

	/**
	 * @brief Register helper functions using the lddw instruction. See
	 * https://docs.kernel.org/bpf/instruction-set.html#id15 for details.
	 * All functions could be null.
	 *
	 * @param[in] map_by_fd A helper to convert a 32-bit file descriptor
	 * into an address of a map
	 * @param[in] map_by_idx A helper to to convert a 32-bit index into an
	 * address of a map
	 * @param[in] map_val Helper to get the address of the first value in a
	 * given map
	 * @param[in] var_addr Helper to get the address of a platform variable
	 * with a given id
	 * @param[in] code_addr Helper to get the address of the instruction at
	 * a specified relative offset in number of (64-bit) instructions
	 */
	virtual void set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
				      uint64_t (*map_by_idx)(uint32_t),
				      uint64_t (*map_val)(uint64_t),
				      uint64_t (*var_addr)(uint32_t),
				      uint64_t (*code_addr)(uint32_t))
	{
		SPDLOG_CRITICAL("Not implemented yet: set_lddw_helpers");
	}

	/**
	 * @brief Optional secret to improve ROP protection.
	 *
	 * @param[in] secret Optional secret to improve ROP protection.
	 * Returns 0 on success, -1 on error (e.g. if the secret is set after
	 * the instructions are loaded).
	 */
	virtual int set_pointer_secret(uint64_t secret)
	{
		SPDLOG_WARN("Not implemented yet: set_pointer_secret");
		return -1;
	}

	/**
	 * @brief Instruct the ebpf runtime to apply unwind-on-success semantics
	 * to a helper function. If the function returns 0, the ebpf runtime
	 * will end execution of the eBPF program and immediately return control
	 * to the caller. This is used for implementing function like the
	 * "bpf_tail_call" helper.
	 *
	 * @param[in] idx Index of the helper function to unwind on success.
	 * @retval 0 Success.
	 * @retval -1 Failure.
	 */
	virtual int set_unwind_function_index(size_t idx)
	{
		SPDLOG_CRITICAL(
			"Not implemented yet: set_unwind_function_index");
		return -1;
	}

	virtual std::vector<uint8_t> do_aot_compile(bool print_ir = false)
	{
		SPDLOG_CRITICAL("Not implemented yet: do_aot_compile");
		return {};
	}

	virtual std::optional<precompiled_ebpf_function>
	load_aot_object(const std::vector<uint8_t> &buf)
	{
		SPDLOG_CRITICAL("Not implemented yet: load_aot_object");
		return {};
	}
};

std::unique_ptr<bpftime_vm_impl> create_vm_instance();

} // namespace bpftime::vm::compat
struct ebpf_vm {
	std::unique_ptr<bpftime::vm::compat::bpftime_vm_impl> vm_instance;
};

#endif
