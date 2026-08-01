#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include <catch2/catch_test_macros.hpp>
#include <linux/filter.h>
#include <unit-test/common_def.hpp>

using namespace bpftime;

static const char *SHM_NAME = "BPFTIME_USER_TO_USER_TAIL_CALL_TEST_SHM";
static const int PROG_ARRAY_MAP_FD = 1001;
static const int TAIL_CALL_TARGET_FD = 1002;

TEST_CASE("Test tail calling from userspace to userspace")
{
	shm_remove remover(SHM_NAME);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	REQUIRE(bpftime_maps_create(PROG_ARRAY_MAP_FD, "prog_array",
				    bpftime::bpf_map_attr{
					    .type = (int)bpftime::bpf_map_type::
						    BPF_MAP_TYPE_PROG_ARRAY,
					    .key_size = 4,
					    .value_size = 4,
					    .max_ents = 4 }) >= 0);

	struct bpf_insn target_insn[] = {
		BPF_MOV64_IMM(0, 0x1234),
		BPF_EXIT_INSN(),
	};
	REQUIRE(bpftime_progs_create(TAIL_CALL_TARGET_FD,
				     (const ebpf_inst *)target_insn,
				     sizeof(target_insn) /
					     sizeof(target_insn[0]),
				     "tail_call_target",
				     (int)bpftime::bpf_prog_type::
					     BPF_PROG_TYPE_SOCKET_FILTER) >= 0);

	{
		int key = 0;
		int value = TAIL_CALL_TARGET_FD;
		REQUIRE(bpftime_map_update_elem(PROG_ARRAY_MAP_FD, &key, &value,
						0) >= 0);
	}

	struct bpf_insn caller_insn[] = {
		BPF_LD_IMM64_RAW_FULL(2, 0, 0, 0, PROG_ARRAY_MAP_FD, 0),
		BPF_MOV64_IMM(3, 0),
		BPF_EMIT_CALL(0x0c),
		BPF_MOV64_IMM(0, 0xdead),
		BPF_EXIT_INSN(),
	};

	bpftime::agent_config config;
	config.set_vm_name("llvm");
	bpftime_prog prog((const ebpf_inst *)caller_insn,
			  sizeof(caller_insn) / sizeof(caller_insn[0]),
			  "tail_call_caller",
			  std::move(config));
	REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&prog) == 0);
	REQUIRE(prog.bpftime_prog_load(false) == 0);

	uint64_t ret = 0;
	int ctx = 0;
	REQUIRE(prog.bpftime_prog_exec(&ctx, sizeof(ctx), &ret) >= 0);
	REQUIRE(ret == 0x1234);

	{
		int key = 0;
		bpftime_close(TAIL_CALL_TARGET_FD);
		REQUIRE(bpftime_map_lookup_elem(PROG_ARRAY_MAP_FD, &key) ==
			nullptr);
	}

	bpftime_destroy_global_shm();
}
