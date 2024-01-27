#include "bpf/libbpf_common.h"
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "linux/bpf.h"
#include "linux/bpf_common.h"
#include <spdlog/spdlog.h>
#include <catch2/catch_test_macros.hpp>
#include <linux/filter.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <bpf_map/userspace/prog_array.hpp>
#include "../common_def.hpp"
using namespace bpftime;

static const char *SHM_NAME = "BPFTIME_PROG_ARRAY_AND_TAIL_CALL_TEST_SHM";
static const int PROG_ARRAY_MAP_FD = 1001;
TEST_CASE("Test tail calling from userspace to kernel")
{
	int err;
	// Create a kernel map
	LIBBPF_OPTS(bpf_map_create_opts, map_create_opts);
	int map_fd = bpf_map_create(BPF_MAP_TYPE_HASH, "simple_map", 4, 4, 1024,
				    &map_create_opts);
	SPDLOG_INFO("Map create err = {}", errno);
	REQUIRE(map_fd >= 0);
	SPDLOG_INFO("Created kernel map fd {}", map_fd);
	char log_buf[1 << 16];
	// Load a program into kernel
	LIBBPF_OPTS(bpf_prog_load_opts, load_opts, .log_level = 1,
		    .log_size = sizeof(log_buf), .log_buf = log_buf);

	// Load two numbers from context, update map[n1]=n2
	struct bpf_insn insns[] = {
		// r2 = *(u32 *)(r1 + 0x0)
		BPF_LDX_MEM(BPF_W, 2, 1, 0),
		// *(u32 *)(r10 - 0x4) = r2
		BPF_STX_MEM(BPF_W, 10, 2, -4),
		// r1 = *(u32 *)(r1 + 0x4)
		BPF_LDX_MEM(BPF_W, 1, 1, 4),
		// *(u32 *)(r10 - 0x8) = r1
		BPF_STX_MEM(BPF_W, 10, 1, -8),
		// r2 = r10
		BPF_MOV64_REG(2, 10),
		// r2 += -0x4
		BPF_ALU64_IMM(BPF_ADD, 2, -0x04),
		// r3 = r10
		BPF_MOV64_REG(3, 10),
		// r3 += -0x8
		BPF_ALU64_IMM(BPF_ADD, 3, -0x08),
		BPF_LD_IMM64_RAW_FULL(1, 1, 0, 0, map_fd, 0),
		// r4 = 0
		BPF_MOV64_IMM(4, 0),
		// call bpf_map_update_elem
		BPF_EMIT_CALL(2),
		// exit
		BPF_EXIT_INSN()
	};
	// Load into kernel
	int prog_fd = bpf_prog_load(BPF_PROG_TYPE_RAW_TRACEPOINT, "test_prog",
				    "GPL", insns, std::size(insns), &load_opts);
	if (prog_fd < 0) {
		SPDLOG_ERROR("Unable to load: \n{}", log_buf);
	}
	REQUIRE(prog_fd >= 0);
	SPDLOG_INFO("Kernel program fd: {}", prog_fd);

	shm_remove remover(SHM_NAME);
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	REQUIRE(bpftime_maps_create(PROG_ARRAY_MAP_FD, "prog_array",
				    bpftime::bpf_map_attr{
					    .type = (int)bpftime::bpf_map_type::
						    BPF_MAP_TYPE_PROG_ARRAY,
					    .key_size = 4,
					    .value_size = 4,
					    .max_ents = 10 }) >= 0);
	{
		int key = 0;
		int value = prog_fd;

		REQUIRE(bpftime_map_update_elem(PROG_ARRAY_MAP_FD, &key, &value,
						0) >= 0);
	}

	struct bpf_insn user_insn[] = {
		// r2 = *(u32 *)(r1 + 0x0)
		BPF_LDX_MEM(BPF_W, 2, 1, 0),
		// *(u32 *)(r10 - 0x40) = r2
		BPF_STX_MEM(BPF_W, 10, 2, -0x40),
		// r1 = *(u32 *)(r1 + 0x4)
		BPF_LDX_MEM(BPF_W, 1, 1, 4),
		// *(u32 *)(r10 - 0x3c) = r1
		BPF_STX_MEM(BPF_W, 10, 1, -0x3c),
		// r1 = r10
		BPF_MOV64_REG(1, 10),
		// r1 += -0x40
		BPF_ALU64_IMM(BPF_ADD, 1, -0x40),
		// r2 = map_ptr
		BPF_LD_IMM64_RAW_FULL(2, 0, 0, 0, 0, PROG_ARRAY_MAP_FD),
		// r3 = 0
		BPF_MOV64_IMM(3, 0),
		// call 0x0c
		BPF_EMIT_CALL(0x0c), BPF_EXIT_INSN()
	};
	bpftime_prog prog((const ebpf_inst *)user_insn, std::size(user_insn),
			  "user_prog");
	REQUIRE(bpftime_helper_group::get_kernel_utils_helper_group()
			.add_helper_group_to_prog(&prog) == 0);
	REQUIRE(prog.bpftime_prog_load(false) == 0);

	uint64_t ret = -1;
	int ctx[2] = { 111, 222 };
	REQUIRE(prog.bpftime_prog_exec((void *)&ctx, sizeof(ctx), &ret) >= 0);

	REQUIRE(ret == 0);
	{
		int key = 111;
		int value;
		REQUIRE(bpf_map_lookup_elem(map_fd, &key, &value) >= 0);
		REQUIRE(value == 222);
	}
	bpftime_destroy_global_shm();
}
