#include <bpftime_prog.hpp>
#include <bpftime_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <bpftime_helper_group.hpp>
static inline bpftime::bpftime_prog *
create_bpftime_prog(const char *name, bpftime::bpftime_object *obj)
{
	using namespace bpftime;
	bpftime_prog *prog = bpftime_object_find_program_by_name(obj, name);
	REQUIRE(prog != nullptr);
	REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(prog) == 0);
	REQUIRE(prog->bpftime_prog_load(false) == 0);
	return prog;
}
