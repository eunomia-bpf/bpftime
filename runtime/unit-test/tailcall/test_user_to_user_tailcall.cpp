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

	bpftime_prog prog((const ebpf_inst *)caller_insn,
			  sizeof(caller_insn) / sizeof(caller_insn[0]),
			  "tail_call_caller");
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
	// Later tests load standalone programs and set their runtime config.
	// destroy_global_shm ends the object's lifetime; leave a valid empty
	// object instead of pointers into the unmapped tailcall segment.
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_NO_CREATE);
}

TEST_CASE("Test tail call copies only the caller context size and zero-fills the rest")
{
	static const int WORD_TARGET_FD = 1003;
	static const int ZERO_TARGET_FD = 1004;
	shm_remove remover("BPFTIME_TAIL_CALL_CTX_BOUNDS_TEST_SHM");
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(bpftime_maps_create(PROG_ARRAY_MAP_FD, "prog_array",
				    bpftime::bpf_map_attr{
					    .type = (int)bpftime::bpf_map_type::
						    BPF_MAP_TYPE_PROG_ARRAY,
					    .key_size = 4,
					    .value_size = 4,
					    .max_ents = 4 }) >= 0);
	// r0 = *(u32 *)(r1 + 0); exit  -- first bytes must survive the copy
	struct bpf_insn word_target_insn[] = {
		BPF_LDX_MEM(BPF_W, 0, 1, 0), BPF_EXIT_INSN(),
	};
	// r0 = *(u32 *)(r1 + 8); exit  -- past a 4-byte caller ctx, must be 0
	struct bpf_insn zero_target_insn[] = {
		BPF_LDX_MEM(BPF_W, 0, 1, 8), BPF_EXIT_INSN(),
	};
	REQUIRE(bpftime_progs_create(WORD_TARGET_FD,
			(const ebpf_inst *)word_target_insn, 2,
			"tail_call_word_target",
			(int)bpftime::bpf_prog_type::
				BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	REQUIRE(bpftime_progs_create(ZERO_TARGET_FD,
			(const ebpf_inst *)zero_target_insn, 2,
			"tail_call_zero_target",
			(int)bpftime::bpf_prog_type::
				BPF_PROG_TYPE_SOCKET_FILTER) >= 0);
	struct bpf_insn caller_insn[] = {
		BPF_LD_IMM64_RAW_FULL(2, 0, 0, 0, PROG_ARRAY_MAP_FD, 0),
		BPF_MOV64_IMM(3, 0), BPF_EMIT_CALL(0x0c),
		BPF_MOV64_IMM(0, 0xdead), BPF_EXIT_INSN(),
	};
	bpftime_prog prog((const ebpf_inst *)caller_insn,
			  sizeof(caller_insn) / sizeof(caller_insn[0]),
			  "tail_call_caller");
	REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&prog) == 0);
	REQUIRE(prog.bpftime_prog_load(false) == 0);

	int key = 0;
	int value = WORD_TARGET_FD;
	REQUIRE(bpftime_map_update_elem(PROG_ARRAY_MAP_FD, &key, &value,
					0) >= 0);
	// Exact 4-byte heap block: ASan poisons every byte past the fourth, so
	// the old unconditional 64-byte memcpy aborts here.
	int *small_ctx = new int(0x2a);
	uint64_t ret = 0;
	REQUIRE(prog.bpftime_prog_exec(small_ctx, sizeof(int), &ret) >= 0);
	REQUIRE(ret == 0x2a);
	delete small_ctx;

	// Repoint at the target reading beyond the caller ctx: the tail of the
	// assembled 64-byte context must be zero-filled.
	value = ZERO_TARGET_FD;
	REQUIRE(bpftime_map_update_elem(PROG_ARRAY_MAP_FD, &key, &value,
					0) >= 0);
	int stack_ctx = 0x2a;
	REQUIRE(prog.bpftime_prog_exec(&stack_ctx, sizeof(stack_ctx), &ret) >=
		0);
	REQUIRE(ret == 0);
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_NO_CREATE);
}
