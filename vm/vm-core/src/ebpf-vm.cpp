#include "ebpf-vm.h"
#include <bpftime_vm_compat.hpp>
#include <cstring>
#include <memory>

extern "C" ebpf_vm *ebpf_create(void)
{
	auto vm = new ebpf_vm;
	vm->vm_instance = bpftime::vm::compat::create_vm_instance();
	return vm;
}

extern "C" void ebpf_destroy(struct ebpf_vm *vm)
{
	delete vm;
}

extern "C" bool ebpf_toggle_bounds_check(struct ebpf_vm *vm, bool enable)
{
	return vm->vm_instance->toggle_bounds_check(enable);
}

extern "C" void
ebpf_set_error_print(struct ebpf_vm *vm,
		     int (*error_printf)(FILE *stream, const char *format, ...))
{
	vm->vm_instance->register_error_print_callback(error_printf);
}

extern "C" int ebpf_register(struct ebpf_vm *vm, unsigned int index,
			     const char *name, void *fn)
{
	return vm->vm_instance->register_external_function(index, name, fn);
}

extern "C" int ebpf_load(struct ebpf_vm *vm, const void *code,
			 uint32_t code_len, char **errmsg)
{
	int err = vm->vm_instance->load_code(code, code_len);
	if (err < 0) {
		*errmsg = strdup(vm->vm_instance->get_error_message().c_str());
	}
	return err;
}

extern "C" void ebpf_unload_code(struct ebpf_vm *vm)
{
	vm->vm_instance->unload_code();
}

extern "C" int ebpf_exec(const struct ebpf_vm *vm, void *mem, size_t mem_len,
			 uint64_t *bpf_return_value)
{
	return vm->vm_instance->exec(mem, mem_len, *bpf_return_value);
}

extern "C" ebpf_jit_fn ebpf_compile(struct ebpf_vm *vm, char **errmsg)
{
	auto func = vm->vm_instance->compile();
	if (!func)
		*errmsg = strdup(vm->vm_instance->get_error_message().c_str());
	return func.value_or(nullptr);
}

extern "C" void ebpf_set_lddw_helpers(struct ebpf_vm *vm,
				      uint64_t (*map_by_fd)(uint32_t),
				      uint64_t (*map_by_idx)(uint32_t),
				      uint64_t (*map_val)(uint64_t),
				      uint64_t (*var_addr)(uint32_t),
				      uint64_t (*code_addr)(uint32_t))
{
	vm->vm_instance->set_lddw_helpers(map_by_fd, map_by_idx, map_val,
					  var_addr, code_addr);
}

extern "C" int ebpf_set_unwind_function_index(struct ebpf_vm *vm,
					      unsigned int idx)
{
	return vm->vm_instance->set_unwind_function_index(idx);
}

extern "C" int ebpf_set_pointer_secret(struct ebpf_vm *vm, uint64_t secret)
{
	return vm->vm_instance->set_pointer_secret(secret);
}

ebpf_jit_fn ebpf_load_aot_object(struct ebpf_vm *vm, const void *buf, size_t buf_len) {
	std::vector<uint8_t> buf_vec((uint8_t *)buf, (uint8_t *)buf + buf_len);
	auto func = vm->vm_instance->load_aot_object(buf_vec);
	return func.value_or(nullptr);
}
