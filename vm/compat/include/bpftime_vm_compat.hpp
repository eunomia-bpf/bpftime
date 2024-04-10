#ifndef _BPFTIME_VM_COMPAT_HPP
#define _BPFTIME_VM_COMPAT_HPP
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace bpftime::vm::compat
{
using precompiled_ebpf_function = uint64_t (*)(void *mem, size_t mem_len);
class bpftime_vm_impl {
    public:
	virtual ~bpftime_vm_impl();
	/**
	 * @brief Get the error message object
	 *
	 * @return std::string
	 */
	virtual std::string get_error_message();
	/**
	 * @brief Toggle whether to enable bounds_check
	 *
	 * @param enable Whether to enable
	 */
	virtual void toggle_bounds_check(bool enable);
	/**
	 * @brief Register a C-style print function for printing error strings
	 *
	 * @param fn The function
	 */
	virtual void
	register_error_print_callback(int (*fn)(FILE *, const char *, ...));
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
					       void *fn);
	/**
	 * @brief Load code into the vm
	 *
	 * @param code Buffer to the code
	 * @param code_len Length of the code, in bytes
	 * @return int
	 */
	virtual int load_code(const void *code, size_t code_len);
	/**
	 * @brief Unload the code
	 *
	 */
	virtual void unload_code();
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
			      uint64_t &bpf_return_value);
	/**
	 * @brief Compile the eBPF program, return the compiled program
	 *
	 * @return Empty optional marks a failure
	 */
	virtual std::optional<precompiled_ebpf_function> compile();
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
				      uint64_t (*code_addr)(uint32_t));
};

std::unique_ptr<bpftime_vm_impl> create_vm_instance();
} // namespace bpftime::vm::compat

#endif
