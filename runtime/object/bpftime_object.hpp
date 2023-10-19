#ifndef _BPFTIME_BPF_OBJECT_H_
#define _BPFTIME_BPF_OBJECT_H_

namespace bpftime
{

// ebpf object file
class bpftime_object;
// open the object elf file and load it into the context
bpftime_object *bpftime_object_open(const char *obj_path);
// load btf associate with the host environment
int bpftime_object_load_relocate_btf(bpftime_object *obj, const char *btf_path);
// close and free the object
void bpftime_object_close(bpftime_object *obj);

// The execution unit or bpf function.
class bpftime_prog;
// find the program by section name
bpftime_prog *bpftime_object_find_program_by_name(bpftime_object *obj,
						  const char *name);
bpftime_prog *bpftime_object_find_program_by_secname(bpftime_object *obj,
						     const char *secname);
bpftime_prog *bpftime_object__next_program(const bpftime_object *obj,
					   bpftime_prog *prog);

#define bpftime_object__for_each_program(pos, obj)                             \
	for ((pos) = bpftime_object__next_program((obj), NULL); (pos) != NULL; \
	     (pos) = bpftime_object__next_program((obj), (pos)))

} // namespace bpftime

#endif // _BPFTIME_BPF_OBJECT_H_
