#include "nv_attach_impl.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <ostream>

static const char *ORIGINAL_TEXT = R"(
.func __memcapture__15 

	// .globl	bpf_main
{
	.local .align 8 .b8 	__local_depot0[16464];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .b16 	%rs<3>;
	.reg .b64 	%rd<9>;

	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.b16 	%rs1, 30828;
	st.local.u16 	[%rd2+16376], %rs1;
	mov.b64 	%rd3, 2675186987289295430;
	st.local.u64 	[%rd2+16368], %rd3;
	mov.b64 	%rd4, 5783296061446517350;
	st.local.u64 	[%rd2+16360], %rd4;
	mov.b64 	%rd5, 2334385650107901261;
	st.local.u64 	[%rd2+16352], %rd5;
	mov.b16 	%rs2, 0;
	st.local.u8 	[%rd2+16378], %rs2;
	add.s64 	%rd6, %rd1, 16352;
	{ // callseq 14, 0
	.param .b64 param0;
	st.param.b64 	[param0], %rd6;
	.param .b64 param1;
	st.param.b64 	[param1], 27;
	.param .b64 param2;
	st.param.b64 	[param2], 10;
	.param .b64 param3;
	st.param.b64 	[param3], 20;
	.param .b64 param4;
	.param .b64 retval0;
	call.uni (retval0), 
	_bpf_helper_ext_0006, 
	(
	param0, 
	param1, 
	param2, 
	param3, 
	param4
	);
	ld.param.b64 	%rd7, [retval0];
	} // callseq 14
	ret;

})";

TEST_CASE("Test register guard")
{
	auto ret = bpftime::attach::add_register_guard_for_ebpf_ptx_func(
		ORIGINAL_TEXT);
	std::cout << ret << std::endl;
	REQUIRE(true);
}
