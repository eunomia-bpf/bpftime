#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <attach/attach_manager/frida_attach_manager.hpp>
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "user/bpftime_driver.hpp"
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif
using namespace bpftime;

#define TEST_INSN_SIZE 12

TEST_CASE("Test daemon driver")
{
	daemon_config cfg;
	char insns[12 * sizeof(ebpf_inst)];
	int pid_0 = 0;
	int pid_1 = 1;
    int id;

	bpftime_driver driver(cfg);
	id = driver.bpftime_progs_create_server(pid_0, 1,
						    (const ebpf_inst *)insns,
						    TEST_INSN_SIZE,
						    "test_prog1", 0);
    REQUIRE(id >= 0);
	id = driver.bpftime_progs_create_server(pid_1, 1,
						    (const ebpf_inst *)insns,
						    TEST_INSN_SIZE,
						    "test_prog2", 0);
    REQUIRE(id >= 0);
}
