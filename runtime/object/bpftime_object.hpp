#ifndef _BPFTIME_BPF_OBJECT_H_
#define _BPFTIME_BPF_OBJECT_H_

#include "bpftime_config.hpp"
namespace bpftime
{

// ebpf object file
class bpftime_object;
// open the object elf file and load it into the context
bpftime_object *bpftime_object_open(const char *obj_path);
bpftime_object *bpftime_object_open(const char *obj_path, runtime_config cfg);
// close and free the object
void bpftime_object_close(bpftime_object *obj);

// The execution unit or bpf function.
class bpftime_prog;
bpftime_prog *bpftime_object_find_program_by_name(bpftime_object *obj,
						  const char *name);
bpftime_prog *bpftime_object__next_program(const bpftime_object *obj,
					   bpftime_prog *prog);

} // namespace bpftime

#endif // _BPFTIME_BPF_OBJECT_H_
