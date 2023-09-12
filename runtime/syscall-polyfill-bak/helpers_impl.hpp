#ifndef _HELPERS_IMPL_H
#define _HELPERS_IMPL_H
#include <ebpf-core.h>
void inject_helpers(ebpf_vm *vm);
extern "C" uint64_t map_ptr_by_fd(uint32_t fd);
extern "C" uint64_t map_val(uint64_t map_ptr);
#endif
