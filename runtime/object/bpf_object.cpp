#if __linux__
#include <linux/bpf.h>
#elif __APPLE__
#include "bpftime_epoll.h"
#endif
#include <string>
#include <list>
#include <memory>
#include <string_view>
#include "bpftime_prog.hpp"
#include <spdlog/spdlog.h>
using namespace std;
using namespace bpftime;
#if __APPLE__
using namespace bpftime_epoll;
#endif

#ifdef BPFTIME_BUILD_WITH_LIBBPF
extern "C" {
#include <bpf/libbpf.h>
#include <bpf/btf.h>
#include <bpf/bpf.h>
}
#endif

namespace bpftime
{

// ebpf object file
class bpftime_object
{
    public:
	bpftime_object(std::string_view path);
	~bpftime_object() = default;
	string obj_path;
	std::unique_ptr<struct bpf_object, decltype(&bpf_object__close)> obj;
	std::unique_ptr<struct btf, decltype(&btf__free)> host_btf;

	std::list<std::unique_ptr<bpftime_prog> > progs;

	void create_programs();
	bpftime_prog *find_program_by_name(std::string_view name) const;
	bpftime_prog *find_program_by_secname(std::string_view name) const;
	bpftime_prog *next_program(bpftime_prog * prog) const;
};

bpftime_prog *bpftime_object::next_program(bpftime_prog *prog) const
{
	if (!prog) {
		if (progs.empty()) {
			return nullptr;
		}
		return progs.front().get();
	}
	for (auto it = progs.begin(); it != progs.end(); ++it) {
		if (it->get() == prog) {
			auto next = std::next(it);
			if (next == progs.end()) {
				return nullptr;
			}
			return next->get();
		}
	}
	return nullptr;
}

void bpftime_object::create_programs()
{
	bpf_program *prog;
	bpf_object__for_each_program(prog, obj.get())
	{
		if (!prog) {
			continue;
		}
		struct ebpf_inst *insns =
			(struct ebpf_inst *)bpf_program__insns(prog);
		size_t cnt = bpf_program__insn_cnt(prog);
		const char *name = bpf_program__name(prog);
		if (!insns || !name) {
			SPDLOG_ERROR("Failed to get insns or name for prog {}",
				     name || "<NULL>");
			continue;
		}
		progs.emplace_back(
			std::make_unique<bpftime_prog>(insns, cnt, name));
	}
}

bpftime_prog *
bpftime_object::find_program_by_secname(std::string_view name) const
{
	const char *sec_name;
	struct bpf_program *prog = NULL;
	bpftime_prog *time_prog = progs.front().get();
	// iterate through the bpftime_prog from prog and bpf_program
	bpf_object__for_each_program(prog, obj.get())
	{
		if (!prog) {
			continue;
		}
		sec_name = bpf_program__section_name(prog);
		if (!sec_name)
			continue;
		if (strcmp(sec_name, name.data()) == 0) {
			return time_prog;
		}
		time_prog = next_program(time_prog);
	}
	return NULL;
}

bpftime_prog *bpftime_object::find_program_by_name(std::string_view name) const
{
	const char *sec_name;
	for (auto &p : progs) {
		sec_name = p->prog_name();
		if (sec_name && strcmp(sec_name, name.data()) == 0) {
			return p.get();
		}
	}
	return NULL;
}

bpftime_object::bpftime_object(std::string_view path)
	: obj_path(path), obj(nullptr, bpf_object__close),
	  host_btf(nullptr, btf__free)
{
	bpf_object *obj_ptr = bpf_object__open(obj_path.data());
	if (!obj_ptr) {
		SPDLOG_ERROR("Failed to open object file {}", obj_path);
		return;
	}
	obj.reset(obj_ptr);
	create_programs();
}

// open the object elf file and load it into the context
bpftime_object *bpftime_object_open(const char *obj_path)
{
	bpftime_object *obj = new bpftime_object(obj_path);
	return obj;
}

// close and free the object
void bpftime_object_close(bpftime_object *obj)
{
	if (!obj) {
		return;
	}
	delete obj;
	return;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
	return vfprintf(stderr, format, args);
#pragma clang diagnostic pop
}

// The execution unit or bpf function.
class bpftime_prog;
// find the program by section name
class bpftime_prog *bpftime_object_find_program_by_name(bpftime_object *obj,
							const char *name)
{
	if (!obj || !name) {
		return NULL;
	}
	return obj->find_program_by_name(name);
}

class bpftime_prog *bpftime_object_find_program_by_secname(bpftime_object *obj,
							   const char *secname)
{
	if (!obj || !secname) {
		return NULL;
	}
	return obj->find_program_by_secname(secname);
}

class bpftime_prog *bpftime_object__next_program(const bpftime_object *obj,
						 class bpftime_prog *prog)
{
	if (!obj) {
		return NULL;
	}
	return obj->next_program(prog);
}

} // namespace bpftime
