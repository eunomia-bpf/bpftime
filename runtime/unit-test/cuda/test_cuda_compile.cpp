
#include "bpf_attach_ctx.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_shm_internal.hpp"
#include "catch2/catch_test_macros.hpp"
#include <bpftime_prog.hpp>
#include <chrono>
#include <iterator>
#include <memory>
#include <thread>
using namespace bpftime;

/**
 * @brief

static void *(* const bpf_map_lookup_elem)(void *map, const void *key) = (void
*) 1;


int my_main(char* mem, unsigned long len) {

    int buf[4]={111,0,0,0};
    int* value = bpf_map_lookup_elem((void*)1,buf);
    ((int*)mem)[0] = *value;

    return 0;
}
 *
 */
static const struct ebpf_inst test_prog[] = {
	// // r3 = 23333
	// { EBPF_OP_MOV64_IMM, 3, 0, 0, 23333 },
	// // *(u32 *)(r1 + 0) = r3
	// { EBPF_OP_STXW, 1, 3, 0, 0 },
	// r6 = r1
	{ EBPF_OP_MOV64_REG, 6, 1, 0, 0 },
	// // r1 = 111
	{ EBPF_OP_MOV64_IMM, 1, 0, 0, 111 },
	// // *(u32 *)(r10 - 16) = r1
	{ EBPF_OP_STXW, 10, 1, -16, 0 },
	// // r1 = 0
	{ EBPF_OP_MOV64_IMM, 1, 0, 0, 0 },
	// // *(u32 *)(r10 - 4) = r1
	{ EBPF_OP_STXW, 10, 1, -4, 0 },
	// // *(u32 *)(r10 - 8) = r1
	{ EBPF_OP_STXW, 10, 1, -8, 0 },
	// // *(u32 *)(r10 - 12) = r1
	{ EBPF_OP_STXW, 10, 1, -12, 0 },
	// r2 = r10
	{ EBPF_OP_MOV64_REG, 2, 10, 0, 0 },
	// r2 += -16
	{ EBPF_OP_ADD64_IMM, 2, 0, 0, -16 },
	// r1 = 1
	{ EBPF_OP_MOV64_IMM, 1, 0, 0, 1 },
	// CALL 1
	{ EBPF_OP_CALL, 0, 0, 0, 1 },
	// r1 = *(u32 *)(r0 + 0)
	{ EBPF_OP_LDXW, 1, 0, 0, 0 },
	// *(u32 *)(r6 + 0) = r1
	{ EBPF_OP_STXW, 6, 1, 0, 0 },
	// r0 = 0
	{ EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
	// EXIT
	{ EBPF_OP_EXIT, 0, 0, 0, 0 }

};

static uint64_t placeholder_helper(uint64_t, uint64_t, uint64_t, uint64_t,
				   uint64_t)
{
	return 0;
}

TEST_CASE("Test cuda compile")
{
	bpftime_prog prog(test_prog, std::size(test_prog),
			  "test_cuda_prog__cuda");
	prog.bpftime_prog_register_raw_helper(
		bpftime_helper_info{ .index = 1,
				     .name = "bpf_map_lookup",
				     .fn = (void *)&placeholder_helper });
	prog.bpftime_prog_load(true);
}

TEST_CASE("CUDA watcher follows attach-context and shm lifetimes")
{
	auto send_request = []() {
		auto *comm = shm_holder.global_shared_memory
				     .get_cuda_comm_shared_mem();
		comm->request_id = -1;
		comm->flag2 = 0;
		comm->flag1 = 1;
		for (int attempt = 0; attempt < 100 && comm->flag2 != 1;
		     attempt++) {
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1));
		}
		return comm->flag2 == 1;
	};

	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	bool request_handled = false;
	auto stale = std::unique_ptr<bpf_attach_ctx>();
	{
		auto first = std::make_unique<bpf_attach_ctx>();
		stale = std::make_unique<bpf_attach_ctx>();
		first.reset();
		request_handled = send_request();
		bpftime_destroy_global_shm();
	}

	bpftime_initialize_global_shm(shm_open_type::SHM_REMOVE_AND_CREATE);
	bool restart_handled = false;
	{
		auto restarted = std::make_unique<bpf_attach_ctx>();
		stale.reset();
		restart_handled = send_request();
	}
	bpftime_destroy_global_shm();
	REQUIRE(request_handled);
	REQUIRE(restart_handled);
}
