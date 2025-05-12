#include "ebpf_inst.h"
#include "nv_attach_impl.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

static const char *ORIGINAL_PTX = R"(

.version 8.1
.target sm_60
.address_size 64

	// .globl	spin_lock               // -- Begin function spin_lock
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.visible .const .align 8 .u64 constData;
.visible .const .align 4 .b8 map_info[4096];
.global .align 1 .b8 _$_str[45] = {107, 101, 114, 110, 101, 108, 32, 102, 117, 110, 99, 116, 105, 111, 110, 32, 101, 110, 116, 101, 114, 101, 100, 44, 32, 109, 101, 109, 61, 37, 108, 120, 44, 32, 109, 101, 109, 115, 122, 61, 37, 108, 100, 10, 0};
.global .align 1 .b8 __const_$_bpf_main_$_buf[16] = {97, 97, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
.global .align 1 .b8 _$_str1[32] = {115, 101, 116, 117, 112, 32, 102, 117, 110, 99, 116, 105, 111, 110, 44, 32, 99, 111, 110, 115, 116, 32, 100, 97, 116, 97, 61, 37, 108, 120, 10, 0};
.global .align 1 .b8 __const_$_bpf_main_$_msg[26] = {77, 101, 115, 115, 97, 103, 101, 32, 102, 114, 111, 109, 32, 98, 112, 102, 58, 32, 37, 100, 44, 32, 37, 108, 120, 0};
.global .align 1 .b8 _$_str2[11] = {99, 97, 108, 108, 32, 100, 111, 110, 101, 10, 0};
.global .align 1 .b8 _$_str3[23] = {103, 111, 116, 32, 114, 101, 115, 112, 111, 110, 115, 101, 32, 37, 100, 32, 97, 116, 32, 37, 100, 10, 0};
                                        // @spin_lock
.visible .func spin_lock(
	.param .b64 spin_lock_param_0
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;

	ld.param.u64 	%rd1, [spin_lock_param_0];
$L__BB0_1:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r1, [%rd1], 0, 1;
	setp.eq.s32 	%p1, %r1, 1;
	@%p1 bra 	$L__BB0_1;
	ret;
                                        // -- End function
}
	// .globl	spin_unlock             // -- Begin function spin_unlock
.visible .func spin_unlock(
	.param .b64 spin_unlock_param_0
)                                       // @spin_unlock
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;

	ld.param.u64 	%rd1, [spin_unlock_param_0];
	atom.exch.b32 	%r1, [%rd1], 0;
	ret;
                                        // -- End function
}
	// .globl	make_helper_call        // -- Begin function make_helper_call
.visible .func  (.param .align 8 .b8 func_retval0[8]) make_helper_call(
	.param .b64 make_helper_call_param_0,
	.param .b32 make_helper_call_param_1
)                                       // @make_helper_call
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<16>;

	ld.param.u32 	%r1, [make_helper_call_param_1];
	ld.param.u64 	%rd6, [make_helper_call_param_0];
	ld.const.u64 	%rd9, [constData];
	// begin inline asm
	mov.u64 %rd7, %globaltimer;
	// end inline asm
	add.s64 	%rd8, %rd9, 8;
$L__BB2_1:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r2, [%rd8], 0, 1;
	setp.eq.s32 	%p1, %r2, 1;
	@%p1 bra 	$L__BB2_1;
	st.u32 	[%rd9+12], %r1;
	st.u64 	[%rd9+16], %rd6;
	add.s64 	%rd10, %rd9, 4;
	mov.u32 	%r3, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd9], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r3, [%rd10];          
	setp.eq.u32 p0, %r3, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd10], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.u64 	%rd3, [%rd9+2147483680];
	atom.exch.b32 	%r4, [%rd8], 0;
	// begin inline asm
	mov.u64 %rd11, %globaltimer;
	// end inline asm
	setp.gt.s32 	%p2, %r1, 7;
	@%p2 bra 	$L__BB2_4;
	sub.s64 	%rd4, %rd11, %rd7;
	mul.wide.s32 	%rd13, %r1, 8;
	add.s64 	%rd14, %rd9, %rd13;
	add.s64 	%rd5, %rd14, 2147483688;
	atom.add.u64 	%rd15, [%rd5], %rd4;
$L__BB2_4:
	st.param.b64 	[func_retval0+0], %rd3;
	ret;
                                        // -- End function
}
	// .globl	_bpf_helper_ext_0001    // -- Begin function _bpf_helper_ext_0001
.visible .func  (.param .b64 func_retval0) _bpf_helper_ext_0001(
	.param .b64 _bpf_helper_ext_0001_param_0,
	.param .b64 _bpf_helper_ext_0001_param_1,
	.param .b64 _bpf_helper_ext_0001_param_2,
	.param .b64 _bpf_helper_ext_0001_param_3,
	.param .b64 _bpf_helper_ext_0001_param_4
)                                       // @_bpf_helper_ext_0001
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<6>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<39>;

	ld.param.u64 	%rd15, [_bpf_helper_ext_0001_param_0];
	ld.const.u64 	%rd28, [constData];
	shr.u64 	%rd17, %rd15, 28;
	and.b64  	%rd18, %rd17, 68719476720;
	mov.u64 	%rd19, map_info;
	add.s64 	%rd20, %rd19, %rd18;
	ld.const.u32 	%r6, [%rd20+4];
	setp.lt.s32 	%p1, %r6, 1;
	@%p1 bra 	$L__BB3_7;
	ld.param.u64 	%rd16, [_bpf_helper_ext_0001_param_1];
	cvt.u64.u32 	%rd3, %r6;
	cvt.u32.u64 	%r8, %rd3;
	and.b32  	%r15, %r8, 3;
	setp.lt.u32 	%p2, %r8, 4;
	mov.u32 	%r14, 0;
	@%p2 bra 	$L__BB3_4;
	add.s64 	%rd2, %rd28, 24;
	and.b64  	%rd4, %rd3, 4294967292;
	add.s64 	%rd5, %rd16, 3;
	mov.u64 	%rd36, 0;
	cvt.u32.u64 	%r9, %rd4;
$L__BB3_3:                              // =>This Inner Loop Header: Depth=1
	add.s64 	%rd22, %rd5, %rd36;
	ld.u8 	%rs1, [%rd22+-3];
	add.s64 	%rd23, %rd2, %rd36;
	st.u8 	[%rd23], %rs1;
	ld.u8 	%rs2, [%rd22+-2];
	st.u8 	[%rd23+1], %rs2;
	ld.u8 	%rs3, [%rd22+-1];
	st.u8 	[%rd23+2], %rs3;
	ld.u8 	%rs4, [%rd22];
	st.u8 	[%rd23+3], %rs4;
	add.s64 	%rd36, %rd36, 4;
	cvt.u32.u64 	%r14, %rd36;
	setp.ne.s32 	%p3, %r9, %r14;
	@%p3 bra 	$L__BB3_3;
$L__BB3_4:
	setp.eq.s32 	%p4, %r15, 0;
	@%p4 bra 	$L__BB3_7;
	cvt.u64.u32 	%rd24, %r14;
	add.s64 	%rd25, %rd24, %rd28;
	add.s64 	%rd38, %rd25, 24;
	add.s64 	%rd37, %rd16, %rd24;
$L__BB3_6:                              // =>This Inner Loop Header: Depth=1
	.pragma "nounroll";
	ld.u8 	%rs5, [%rd37];
	st.u8 	[%rd38], %rs5;
	add.s64 	%rd38, %rd38, 1;
	add.s64 	%rd37, %rd37, 1;
	add.s32 	%r15, %r15, -1;
	setp.ne.s32 	%p5, %r15, 0;
	@%p5 bra 	$L__BB3_6;
$L__BB3_7:
	// begin inline asm
	mov.u64 %rd26, %globaltimer;
	// end inline asm
	add.s64 	%rd27, %rd28, 8;
$L__BB3_8:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r10, [%rd27], 0, 1;
	setp.eq.s32 	%p6, %r10, 1;
	@%p6 bra 	$L__BB3_8;
	mov.u32 	%r12, 1;
	st.u32 	[%rd28+12], %r12;
	st.u64 	[%rd28+16], %rd15;
	add.s64 	%rd29, %rd28, 4;
	mov.u32 	%r11, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd28], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r11, [%rd29];          
	setp.eq.u32 p0, %r11, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd29], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.u64 	%rd31, [%rd28+2147483680];
	atom.exch.b32 	%r13, [%rd27], 0;
	// begin inline asm
	mov.u64 %rd30, %globaltimer;
	// end inline asm
	sub.s64 	%rd33, %rd30, %rd26;
	add.s64 	%rd34, %rd28, 2147483696;
	atom.add.u64 	%rd35, [%rd34], %rd33;
	st.param.b64 	[func_retval0+0], %rd31;
	ret;
                                        // -- End function
}
	// .globl	_bpf_helper_ext_0002    // -- Begin function _bpf_helper_ext_0002
.visible .func  (.param .b64 func_retval0) _bpf_helper_ext_0002(
	.param .b64 _bpf_helper_ext_0002_param_0,
	.param .b64 _bpf_helper_ext_0002_param_1,
	.param .b64 _bpf_helper_ext_0002_param_2,
	.param .b64 _bpf_helper_ext_0002_param_3,
	.param .b64 _bpf_helper_ext_0002_param_4
)                                       // @_bpf_helper_ext_0002
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<11>;
	.reg .b32 	%r<28>;
	.reg .b64 	%rd<64>;

	ld.param.u64 	%rd28, [_bpf_helper_ext_0002_param_0];
	ld.const.u64 	%rd50, [constData];
	shr.u64 	%rd3, %rd28, 32;
	shl.b64 	%rd32, %rd3, 4;
	mov.u64 	%rd33, map_info;
	add.s64 	%rd34, %rd33, %rd32;
	ld.const.u32 	%r11, [%rd34+4];
	setp.lt.s32 	%p1, %r11, 1;
	@%p1 bra 	$L__BB4_7;
	ld.param.u64 	%rd29, [_bpf_helper_ext_0002_param_1];
	cvt.u64.u32 	%rd4, %r11;
	cvt.u32.u64 	%r13, %rd4;
	and.b32  	%r25, %r13, 3;
	setp.lt.u32 	%p2, %r13, 4;
	mov.u32 	%r24, 0;
	@%p2 bra 	$L__BB4_4;
	add.s64 	%rd2, %rd50, 24;
	and.b64  	%rd5, %rd4, 4294967292;
	add.s64 	%rd6, %rd29, 3;
	mov.u64 	%rd58, 0;
	cvt.u32.u64 	%r14, %rd5;
$L__BB4_3:                              // =>This Inner Loop Header: Depth=1
	add.s64 	%rd36, %rd6, %rd58;
	ld.u8 	%rs1, [%rd36+-3];
	add.s64 	%rd37, %rd2, %rd58;
	st.u8 	[%rd37], %rs1;
	ld.u8 	%rs2, [%rd36+-2];
	st.u8 	[%rd37+1], %rs2;
	ld.u8 	%rs3, [%rd36+-1];
	st.u8 	[%rd37+2], %rs3;
	ld.u8 	%rs4, [%rd36];
	st.u8 	[%rd37+3], %rs4;
	add.s64 	%rd58, %rd58, 4;
	cvt.u32.u64 	%r24, %rd58;
	setp.ne.s32 	%p3, %r14, %r24;
	@%p3 bra 	$L__BB4_3;
$L__BB4_4:
	setp.eq.s32 	%p4, %r25, 0;
	@%p4 bra 	$L__BB4_7;
	cvt.u64.u32 	%rd38, %r24;
	add.s64 	%rd39, %rd38, %rd50;
	add.s64 	%rd60, %rd39, 24;
	add.s64 	%rd59, %rd29, %rd38;
$L__BB4_6:                              // =>This Inner Loop Header: Depth=1
	.pragma "nounroll";
	ld.u8 	%rs5, [%rd59];
	st.u8 	[%rd60], %rs5;
	add.s64 	%rd60, %rd60, 1;
	add.s64 	%rd59, %rd59, 1;
	add.s32 	%r25, %r25, -1;
	setp.ne.s32 	%p5, %r25, 0;
	@%p5 bra 	$L__BB4_6;
$L__BB4_7:
	ld.param.u64 	%rd31, [_bpf_helper_ext_0002_param_3];
	ld.const.u32 	%r15, [%rd34+8];
	setp.lt.s32 	%p6, %r15, 1;
	@%p6 bra 	$L__BB4_14;
	ld.param.u64 	%rd30, [_bpf_helper_ext_0002_param_2];
	mov.u32 	%r26, 0;
	cvt.u64.u32 	%rd16, %r15;
	cvt.u32.u64 	%r17, %rd16;
	and.b32  	%r27, %r17, 3;
	setp.lt.u32 	%p7, %r17, 4;
	@%p7 bra 	$L__BB4_11;
	add.s64 	%rd15, %rd50, 1073741848;
	and.b64  	%rd17, %rd16, 4294967292;
	add.s64 	%rd18, %rd30, 3;
	mov.u64 	%rd61, 0;
	cvt.u32.u64 	%r18, %rd17;
$L__BB4_10:                             // =>This Inner Loop Header: Depth=1
	add.s64 	%rd44, %rd18, %rd61;
	ld.u8 	%rs6, [%rd44+-3];
	add.s64 	%rd45, %rd15, %rd61;
	st.u8 	[%rd45], %rs6;
	ld.u8 	%rs7, [%rd44+-2];
	st.u8 	[%rd45+1], %rs7;
	ld.u8 	%rs8, [%rd44+-1];
	st.u8 	[%rd45+2], %rs8;
	ld.u8 	%rs9, [%rd44];
	st.u8 	[%rd45+3], %rs9;
	add.s64 	%rd61, %rd61, 4;
	cvt.u32.u64 	%r26, %rd61;
	setp.ne.s32 	%p8, %r18, %r26;
	@%p8 bra 	$L__BB4_10;
$L__BB4_11:
	setp.eq.s32 	%p9, %r27, 0;
	@%p9 bra 	$L__BB4_14;
	cvt.u64.u32 	%rd46, %r26;
	add.s64 	%rd47, %rd46, %rd50;
	add.s64 	%rd63, %rd47, 1073741848;
	add.s64 	%rd62, %rd30, %rd46;
$L__BB4_13:                             // =>This Inner Loop Header: Depth=1
	.pragma "nounroll";
	ld.u8 	%rs10, [%rd62];
	st.u8 	[%rd63], %rs10;
	add.s64 	%rd63, %rd63, 1;
	add.s64 	%rd62, %rd62, 1;
	add.s32 	%r27, %r27, -1;
	setp.ne.s32 	%p10, %r27, 0;
	@%p10 bra 	$L__BB4_13;
$L__BB4_14:
	st.u64 	[%rd50+2147483672], %rd31;
	// begin inline asm
	mov.u64 %rd48, %globaltimer;
	// end inline asm
	add.s64 	%rd49, %rd50, 8;
$L__BB4_15:                             // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r19, [%rd49], 0, 1;
	setp.eq.s32 	%p11, %r19, 1;
	@%p11 bra 	$L__BB4_15;
	mov.u32 	%r21, 2;
	st.u32 	[%rd50+12], %r21;
	st.u64 	[%rd50+16], %rd28;
	add.s64 	%rd51, %rd50, 4;
	mov.u32 	%r20, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd50], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r20, [%rd51];          
	setp.eq.u32 p0, %r20, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd51], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.s32 	%rd53, [%rd50+2147483680];
	atom.exch.b32 	%r22, [%rd49], 0;
	// begin inline asm
	mov.u64 %rd52, %globaltimer;
	// end inline asm
	sub.s64 	%rd55, %rd52, %rd48;
	add.s64 	%rd56, %rd50, 2147483704;
	atom.add.u64 	%rd57, [%rd56], %rd55;
	st.param.b64 	[func_retval0+0], %rd53;
	ret;
                                        // -- End function
}
	// .globl	_bpf_helper_ext_0003    // -- Begin function _bpf_helper_ext_0003
.visible .func  (.param .b64 func_retval0) _bpf_helper_ext_0003(
	.param .b64 _bpf_helper_ext_0003_param_0,
	.param .b64 _bpf_helper_ext_0003_param_1,
	.param .b64 _bpf_helper_ext_0003_param_2,
	.param .b64 _bpf_helper_ext_0003_param_3,
	.param .b64 _bpf_helper_ext_0003_param_4
)                                       // @_bpf_helper_ext_0003
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<6>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<39>;

	ld.param.u64 	%rd15, [_bpf_helper_ext_0003_param_0];
	ld.const.u64 	%rd28, [constData];
	shr.u64 	%rd17, %rd15, 28;
	and.b64  	%rd18, %rd17, 68719476720;
	mov.u64 	%rd19, map_info;
	add.s64 	%rd20, %rd19, %rd18;
	ld.const.u32 	%r6, [%rd20+4];
	setp.lt.s32 	%p1, %r6, 1;
	@%p1 bra 	$L__BB5_7;
	ld.param.u64 	%rd16, [_bpf_helper_ext_0003_param_1];
	cvt.u64.u32 	%rd3, %r6;
	cvt.u32.u64 	%r8, %rd3;
	and.b32  	%r15, %r8, 3;
	setp.lt.u32 	%p2, %r8, 4;
	mov.u32 	%r14, 0;
	@%p2 bra 	$L__BB5_4;
	add.s64 	%rd2, %rd28, 24;
	and.b64  	%rd4, %rd3, 4294967292;
	add.s64 	%rd5, %rd16, 3;
	mov.u64 	%rd36, 0;
	cvt.u32.u64 	%r9, %rd4;
$L__BB5_3:                              // =>This Inner Loop Header: Depth=1
	add.s64 	%rd22, %rd5, %rd36;
	ld.u8 	%rs1, [%rd22+-3];
	add.s64 	%rd23, %rd2, %rd36;
	st.u8 	[%rd23], %rs1;
	ld.u8 	%rs2, [%rd22+-2];
	st.u8 	[%rd23+1], %rs2;
	ld.u8 	%rs3, [%rd22+-1];
	st.u8 	[%rd23+2], %rs3;
	ld.u8 	%rs4, [%rd22];
	st.u8 	[%rd23+3], %rs4;
	add.s64 	%rd36, %rd36, 4;
	cvt.u32.u64 	%r14, %rd36;
	setp.ne.s32 	%p3, %r9, %r14;
	@%p3 bra 	$L__BB5_3;
$L__BB5_4:
	setp.eq.s32 	%p4, %r15, 0;
	@%p4 bra 	$L__BB5_7;
	cvt.u64.u32 	%rd24, %r14;
	add.s64 	%rd25, %rd24, %rd28;
	add.s64 	%rd38, %rd25, 24;
	add.s64 	%rd37, %rd16, %rd24;
$L__BB5_6:                              // =>This Inner Loop Header: Depth=1
	.pragma "nounroll";
	ld.u8 	%rs5, [%rd37];
	st.u8 	[%rd38], %rs5;
	add.s64 	%rd38, %rd38, 1;
	add.s64 	%rd37, %rd37, 1;
	add.s32 	%r15, %r15, -1;
	setp.ne.s32 	%p5, %r15, 0;
	@%p5 bra 	$L__BB5_6;
$L__BB5_7:
	// begin inline asm
	mov.u64 %rd26, %globaltimer;
	// end inline asm
	add.s64 	%rd27, %rd28, 8;
$L__BB5_8:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r10, [%rd27], 0, 1;
	setp.eq.s32 	%p6, %r10, 1;
	@%p6 bra 	$L__BB5_8;
	mov.u32 	%r12, 3;
	st.u32 	[%rd28+12], %r12;
	st.u64 	[%rd28+16], %rd15;
	add.s64 	%rd29, %rd28, 4;
	mov.u32 	%r11, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd28], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r11, [%rd29];          
	setp.eq.u32 p0, %r11, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd29], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.s32 	%rd31, [%rd28+2147483680];
	atom.exch.b32 	%r13, [%rd27], 0;
	// begin inline asm
	mov.u64 %rd30, %globaltimer;
	// end inline asm
	sub.s64 	%rd33, %rd30, %rd26;
	add.s64 	%rd34, %rd28, 2147483712;
	atom.add.u64 	%rd35, [%rd34], %rd33;
	st.param.b64 	[func_retval0+0], %rd31;
	ret;
                                        // -- End function
}
	// .globl	_bpf_helper_ext_0006    // -- Begin function _bpf_helper_ext_0006
.visible .func  (.param .b64 func_retval0) _bpf_helper_ext_0006(
	.param .b64 _bpf_helper_ext_0006_param_0,
	.param .b64 _bpf_helper_ext_0006_param_1,
	.param .b64 _bpf_helper_ext_0006_param_2,
	.param .b64 _bpf_helper_ext_0006_param_3,
	.param .b64 _bpf_helper_ext_0006_param_4
)                                       // @_bpf_helper_ext_0006
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<28>;

	ld.param.u64 	%rd10, [_bpf_helper_ext_0006_param_4];
	ld.param.u64 	%rd9, [_bpf_helper_ext_0006_param_3];
	ld.param.u64 	%rd8, [_bpf_helper_ext_0006_param_2];
	ld.param.u64 	%rd7, [_bpf_helper_ext_0006_param_1];
	ld.const.u64 	%rd1, [constData];
	setp.eq.s64 	%p1, %rd7, 0;
	@%p1 bra 	$L__BB6_3;
	ld.param.u64 	%rd6, [_bpf_helper_ext_0006_param_0];
	add.s64 	%rd2, %rd1, 24;
	mov.u64 	%rd27, 0;
$L__BB6_2:                              // =>This Inner Loop Header: Depth=1
	add.s64 	%rd12, %rd6, %rd27;
	ld.u8 	%rs1, [%rd12];
	add.s64 	%rd13, %rd2, %rd27;
	st.u8 	[%rd13], %rs1;
	add.s64 	%rd14, %rd27, 1;
	and.b64  	%rd27, %rd14, 4294967295;
	setp.lt.u64 	%p2, %rd27, %rd7;
	@%p2 bra 	$L__BB6_2;
$L__BB6_3:
	st.u32 	[%rd1+1024], %rd7;
	st.u64 	[%rd1+1032], %rd8;
	st.u64 	[%rd1+1040], %rd9;
	st.u64 	[%rd1+1048], %rd10;
	// begin inline asm
	mov.u64 %rd15, %globaltimer;
	// end inline asm
	add.s64 	%rd16, %rd1, 8;
$L__BB6_4:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r1, [%rd16], 0, 1;
	setp.eq.s32 	%p3, %r1, 1;
	@%p3 bra 	$L__BB6_4;
	mov.u32 	%r3, 6;
	st.u32 	[%rd1+12], %r3;
	mov.u64 	%rd20, 0;
	st.u64 	[%rd1+16], %rd20;
	add.s64 	%rd18, %rd1, 4;
	mov.u32 	%r2, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd1], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r2, [%rd18];          
	setp.eq.u32 p0, %r2, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd18], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.s32 	%rd21, [%rd1+2147483680];
	atom.exch.b32 	%r4, [%rd16], 0;
	// begin inline asm
	mov.u64 %rd19, %globaltimer;
	// end inline asm
	sub.s64 	%rd23, %rd19, %rd15;
	add.s64 	%rd24, %rd1, 2147483736;
	atom.add.u64 	%rd25, [%rd24], %rd23;
	st.param.b64 	[func_retval0+0], %rd21;
	ret;
                                        // -- End function
}
	// .globl	_bpf_helper_ext_0501    // -- Begin function _bpf_helper_ext_0501
.visible .func  (.param .b64 func_retval0) _bpf_helper_ext_0501(
	.param .b64 _bpf_helper_ext_0501_param_0,
	.param .b64 _bpf_helper_ext_0501_param_1,
	.param .b64 _bpf_helper_ext_0501_param_2,
	.param .b64 _bpf_helper_ext_0501_param_3,
	.param .b64 _bpf_helper_ext_0501_param_4
)                                       // @_bpf_helper_ext_0501
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<6>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<23>;

	ld.param.u64 	%rd7, [_bpf_helper_ext_0501_param_0];
	ld.const.u64 	%rd15, [constData];
	add.s64 	%rd2, %rd15, 24;
	ld.u8 	%rs5, [%rd7];
	setp.eq.s16 	%p1, %rs5, 0;
	mov.u64 	%rd22, 0;
	@%p1 bra 	$L__BB7_3;
	add.s64 	%rd3, %rd7, 1;
	mov.u64 	%rd22, 0;
$L__BB7_2:                              // =>This Inner Loop Header: Depth=1
	add.s64 	%rd10, %rd2, %rd22;
	st.u8 	[%rd10], %rs5;
	add.s64 	%rd11, %rd3, %rd22;
	ld.u8 	%rs5, [%rd11];
	setp.ne.s16 	%p2, %rs5, 0;
	add.s64 	%rd22, %rd22, 1;
	@%p2 bra 	$L__BB7_2;
$L__BB7_3:
	add.s64 	%rd13, %rd2, %rd22;
	mov.u16 	%rs4, 0;
	st.u8 	[%rd13], %rs4;
	// begin inline asm
	mov.u64 %rd12, %globaltimer;
	// end inline asm
	add.s64 	%rd14, %rd15, 8;
$L__BB7_4:                              // =>This Inner Loop Header: Depth=1
	atom.cas.b32 	%r1, [%rd14], 0, 1;
	setp.eq.s32 	%p3, %r1, 1;
	@%p3 bra 	$L__BB7_4;
	mov.u32 	%r3, 501;
	st.u32 	[%rd15+12], %r3;
	mov.u64 	%rd18, 0;
	st.u64 	[%rd15+16], %rd18;
	add.s64 	%rd16, %rd15, 4;
	mov.u32 	%r2, 42;
	// begin inline asm
	.reg .pred p0;                   
	membar.sys;                      
	st.global.u32 [%rd15], 1;           
	spin_wait:                       
	membar.sys;                      
	ld.global.u32 %r2, [%rd16];          
	setp.eq.u32 p0, %r2, 0;           
	@p0 bra spin_wait;               
	st.global.u32 [%rd16], 0;           
	membar.sys;                      
	
	// end inline asm
	ld.s32 	%rd19, [%rd15+2147483680];
	atom.exch.b32 	%r4, [%rd14], 0;
	// begin inline asm
	mov.u64 %rd17, %globaltimer;
	// end inline asm
	st.param.b64 	[func_retval0+0], %rd19;
	ret;
                                        // -- End function
}
.func __memcapture__1 
	// .globl	bpf_main
{
	.local .align 8 .b8 	__local_depot0[16464];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .b64 	%rd<14>;
	.local .align 8 .b8 	__ptx_register_save_area[112];
	.reg .b64 %rd_ptx_instr_base;
	.reg .b64 %rd_ptx_instr_addr;
	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	// --- BEGIN REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Use tempBaseReg for the save area's SPL
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	st.local.u64 [%rd_ptx_instr_addr], %rd0;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	st.local.u64 [%rd_ptx_instr_addr], %rd1;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	st.local.u64 [%rd_ptx_instr_addr], %rd2;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	st.local.u64 [%rd_ptx_instr_addr], %rd3;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	st.local.u64 [%rd_ptx_instr_addr], %rd4;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	st.local.u64 [%rd_ptx_instr_addr], %rd5;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	st.local.u64 [%rd_ptx_instr_addr], %rd6;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	st.local.u64 [%rd_ptx_instr_addr], %rd7;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	st.local.u64 [%rd_ptx_instr_addr], %rd8;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	st.local.u64 [%rd_ptx_instr_addr], %rd9;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	st.local.u64 [%rd_ptx_instr_addr], %rd10;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	st.local.u64 [%rd_ptx_instr_addr], %rd11;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	st.local.u64 [%rd_ptx_instr_addr], %rd12;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	st.local.u64 [%rd_ptx_instr_addr], %rd13;
	// --- END REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.b64 	%rd3, 0;
	st.local.u64 	[%rd2+16376], %rd3;
	add.s64 	%rd4, %rd1, 16328;
	mov.b64 	%rd5, 7881706585694561388;
	st.u64 	[%SP+16328], %rd5;
	mov.b64 	%rd6, 7237888371382973742;
	st.u64 	[%SP+16336], %rd6;
	mov.b64 	%rd7, 3544713747314519093;
	st.u64 	[%SP+16344], %rd7;
	mov.b64 	%rd8, 7589810523092574573;
	st.u64 	[%SP+16352], %rd8;
	mov.b64 	%rd9, 3482239410544862572;
	st.u64 	[%SP+16360], %rd9;
	mov.b64 	%rd10, 7021781904288796767;
	st.u64 	[%SP+16368], %rd10;
	mov.b64 	%rd11, 254966521709;
	st.u64 	[%SP+16376], %rd11;
	{ // callseq 0, 0
	.param .b64 param0;
	st.param.b64 	[param0], %rd4;
	.param .b64 param1;
	st.param.b64 	[param1], 254966521709;
	.param .b64 param2;
	.param .b64 param3;
	.param .b64 param4;
	.param .b64 retval0;
	call.uni (retval0), 
	_bpf_helper_ext_0501, 
	(
	param0, 
	param1, 
	param2, 
	param3, 
	param4
	);
	ld.param.b64 	%rd12, [retval0];
	} // callseq 0

	// --- BEGIN REGISTER RESTORING (POP from __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Base for restoring
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	ld.local.u64 %rd13, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	ld.local.u64 %rd12, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	ld.local.u64 %rd11, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	ld.local.u64 %rd10, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	ld.local.u64 %rd9, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	ld.local.u64 %rd8, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	ld.local.u64 %rd7, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	ld.local.u64 %rd6, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	ld.local.u64 %rd5, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	ld.local.u64 %rd4, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	ld.local.u64 %rd3, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	ld.local.u64 %rd2, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	ld.local.u64 %rd1, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	ld.local.u64 %rd0, [%rd_ptx_instr_addr];
	// --- END REGISTER RESTORING (POP from __ptx_register_save_area) ---
	ret;
}

.func __memcapture__2 
	// .globl	bpf_main
{
	.local .align 8 .b8 	__local_depot0[16464];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .b64 	%rd<14>;
	.local .align 8 .b8 	__ptx_register_save_area[112];
	.reg .b64 %rd_ptx_instr_base;
	.reg .b64 %rd_ptx_instr_addr;
	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	// --- BEGIN REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Use tempBaseReg for the save area's SPL
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	st.local.u64 [%rd_ptx_instr_addr], %rd0;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	st.local.u64 [%rd_ptx_instr_addr], %rd1;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	st.local.u64 [%rd_ptx_instr_addr], %rd2;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	st.local.u64 [%rd_ptx_instr_addr], %rd3;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	st.local.u64 [%rd_ptx_instr_addr], %rd4;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	st.local.u64 [%rd_ptx_instr_addr], %rd5;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	st.local.u64 [%rd_ptx_instr_addr], %rd6;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	st.local.u64 [%rd_ptx_instr_addr], %rd7;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	st.local.u64 [%rd_ptx_instr_addr], %rd8;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	st.local.u64 [%rd_ptx_instr_addr], %rd9;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	st.local.u64 [%rd_ptx_instr_addr], %rd10;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	st.local.u64 [%rd_ptx_instr_addr], %rd11;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	st.local.u64 [%rd_ptx_instr_addr], %rd12;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	st.local.u64 [%rd_ptx_instr_addr], %rd13;
	// --- END REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.b64 	%rd3, 0;
	st.local.u64 	[%rd2+16376], %rd3;
	add.s64 	%rd4, %rd1, 16328;
	mov.b64 	%rd5, 7881706585694561388;
	st.u64 	[%SP+16328], %rd5;
	mov.b64 	%rd6, 7237888371382973742;
	st.u64 	[%SP+16336], %rd6;
	mov.b64 	%rd7, 3544713747314519094;
	st.u64 	[%SP+16344], %rd7;
	mov.b64 	%rd8, 7589810523092574573;
	st.u64 	[%SP+16352], %rd8;
	mov.b64 	%rd9, 3482239410544862572;
	st.u64 	[%SP+16360], %rd9;
	mov.b64 	%rd10, 7021781904288796767;
	st.u64 	[%SP+16368], %rd10;
	mov.b64 	%rd11, 254966587245;
	st.u64 	[%SP+16376], %rd11;
	{ // callseq 1, 0
	.param .b64 param0;
	st.param.b64 	[param0], %rd4;
	.param .b64 param1;
	st.param.b64 	[param1], 254966587245;
	.param .b64 param2;
	.param .b64 param3;
	.param .b64 param4;
	.param .b64 retval0;
	call.uni (retval0), 
	_bpf_helper_ext_0501, 
	(
	param0, 
	param1, 
	param2, 
	param3, 
	param4
	);
	ld.param.b64 	%rd12, [retval0];
	} // callseq 1

	// --- BEGIN REGISTER RESTORING (POP from __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Base for restoring
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	ld.local.u64 %rd13, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	ld.local.u64 %rd12, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	ld.local.u64 %rd11, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	ld.local.u64 %rd10, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	ld.local.u64 %rd9, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	ld.local.u64 %rd8, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	ld.local.u64 %rd7, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	ld.local.u64 %rd6, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	ld.local.u64 %rd5, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	ld.local.u64 %rd4, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	ld.local.u64 %rd3, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	ld.local.u64 %rd2, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	ld.local.u64 %rd1, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	ld.local.u64 %rd0, [%rd_ptx_instr_addr];
	// --- END REGISTER RESTORING (POP from __ptx_register_save_area) ---
	ret;
}

.func __memcapture__3 
	// .globl	bpf_main
{
	.local .align 8 .b8 	__local_depot0[16464];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .b64 	%rd<14>;
	.local .align 8 .b8 	__ptx_register_save_area[112];
	.reg .b64 %rd_ptx_instr_base;
	.reg .b64 %rd_ptx_instr_addr;
	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	// --- BEGIN REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Use tempBaseReg for the save area's SPL
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	st.local.u64 [%rd_ptx_instr_addr], %rd0;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	st.local.u64 [%rd_ptx_instr_addr], %rd1;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	st.local.u64 [%rd_ptx_instr_addr], %rd2;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	st.local.u64 [%rd_ptx_instr_addr], %rd3;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	st.local.u64 [%rd_ptx_instr_addr], %rd4;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	st.local.u64 [%rd_ptx_instr_addr], %rd5;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	st.local.u64 [%rd_ptx_instr_addr], %rd6;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	st.local.u64 [%rd_ptx_instr_addr], %rd7;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	st.local.u64 [%rd_ptx_instr_addr], %rd8;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	st.local.u64 [%rd_ptx_instr_addr], %rd9;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	st.local.u64 [%rd_ptx_instr_addr], %rd10;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	st.local.u64 [%rd_ptx_instr_addr], %rd11;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	st.local.u64 [%rd_ptx_instr_addr], %rd12;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	st.local.u64 [%rd_ptx_instr_addr], %rd13;
	// --- END REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.b64 	%rd3, 0;
	st.local.u64 	[%rd2+16376], %rd3;
	add.s64 	%rd4, %rd1, 16328;
	mov.b64 	%rd5, 7881706585694561388;
	st.u64 	[%SP+16328], %rd5;
	mov.b64 	%rd6, 7237888371382973742;
	st.u64 	[%SP+16336], %rd6;
	mov.b64 	%rd7, 3544713747314519095;
	st.u64 	[%SP+16344], %rd7;
	mov.b64 	%rd8, 7589810523092574573;
	st.u64 	[%SP+16352], %rd8;
	mov.b64 	%rd9, 3482239410544862572;
	st.u64 	[%SP+16360], %rd9;
	mov.b64 	%rd10, 7021781904288796767;
	st.u64 	[%SP+16368], %rd10;
	mov.b64 	%rd11, 254966652781;
	st.u64 	[%SP+16376], %rd11;
	{ // callseq 2, 0
	.param .b64 param0;
	st.param.b64 	[param0], %rd4;
	.param .b64 param1;
	st.param.b64 	[param1], 254966652781;
	.param .b64 param2;
	.param .b64 param3;
	.param .b64 param4;
	.param .b64 retval0;
	call.uni (retval0), 
	_bpf_helper_ext_0501, 
	(
	param0, 
	param1, 
	param2, 
	param3, 
	param4
	);
	ld.param.b64 	%rd12, [retval0];
	} // callseq 2

	// --- BEGIN REGISTER RESTORING (POP from __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Base for restoring
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 104;
	ld.local.u64 %rd13, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 96;
	ld.local.u64 %rd12, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 88;
	ld.local.u64 %rd11, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 80;
	ld.local.u64 %rd10, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	ld.local.u64 %rd9, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	ld.local.u64 %rd8, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	ld.local.u64 %rd7, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	ld.local.u64 %rd6, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	ld.local.u64 %rd5, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	ld.local.u64 %rd4, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	ld.local.u64 %rd3, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	ld.local.u64 %rd2, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	ld.local.u64 %rd1, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	ld.local.u64 %rd0, [%rd_ptx_instr_addr];
	// --- END REGISTER RESTORING (POP from __ptx_register_save_area) ---
	ret;
}

.func __memcapture__4 
	// .globl	bpf_main
{
	.local .align 8 .b8 	__local_depot0[16464];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .b64 	%rd<10>;
	.local .align 8 .b8 	__ptx_register_save_area[80];
	.reg .b64 %rd_ptx_instr_base;
	.reg .b64 %rd_ptx_instr_addr;
	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	// --- BEGIN REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Use tempBaseReg for the save area's SPL
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	st.local.u64 [%rd_ptx_instr_addr], %rd0;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	st.local.u64 [%rd_ptx_instr_addr], %rd1;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	st.local.u64 [%rd_ptx_instr_addr], %rd2;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	st.local.u64 [%rd_ptx_instr_addr], %rd3;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	st.local.u64 [%rd_ptx_instr_addr], %rd4;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	st.local.u64 [%rd_ptx_instr_addr], %rd5;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	st.local.u64 [%rd_ptx_instr_addr], %rd6;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	st.local.u64 [%rd_ptx_instr_addr], %rd7;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	st.local.u64 [%rd_ptx_instr_addr], %rd8;
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	st.local.u64 [%rd_ptx_instr_addr], %rd9;
	// --- END REGISTER SAVING (PUSH to __ptx_register_save_area) ---
	add.u64 	%rd1, %SP, 0;
	add.u64 	%rd2, %SPL, 0;
	mov.b64 	%rd3, 0;
	st.local.u64 	[%rd2+16376], %rd3;
	add.s64 	%rd4, %rd1, 16352;
	mov.b64 	%rd5, 8391171955405775980;
	st.u64 	[%SP+16352], %rd5;
	mov.b64 	%rd6, 3923239045604537646;
	st.u64 	[%SP+16360], %rd6;
	mov.b64 	%rd7, 4277661392684392492;
	st.u64 	[%SP+16368], %rd7;
	{ // callseq 3, 0
	.param .b64 param0;
	st.param.b64 	[param0], %rd4;
	.param .b64 param1;
	st.param.b64 	[param1], 4277661392684392492;
	.param .b64 param2;
	.param .b64 param3;
	.param .b64 param4;
	.param .b64 retval0;
	call.uni (retval0), 
	_bpf_helper_ext_0501, 
	(
	param0, 
	param1, 
	param2, 
	param3, 
	param4
	);
	ld.param.b64 	%rd8, [retval0];
	} // callseq 3

	// --- BEGIN REGISTER RESTORING (POP from __ptx_register_save_area) ---
	mov.u64 %rd_ptx_instr_base, __ptx_register_save_area; // Base for restoring
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 72;
	ld.local.u64 %rd9, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 64;
	ld.local.u64 %rd8, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 56;
	ld.local.u64 %rd7, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 48;
	ld.local.u64 %rd6, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 40;
	ld.local.u64 %rd5, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 32;
	ld.local.u64 %rd4, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 24;
	ld.local.u64 %rd3, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 16;
	ld.local.u64 %rd2, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 8;
	ld.local.u64 %rd1, [%rd_ptx_instr_addr];
	add.u64 %rd_ptx_instr_addr, %rd_ptx_instr_base, 0;
	ld.local.u64 %rd0, [%rd_ptx_instr_addr];
	// --- END REGISTER RESTORING (POP from __ptx_register_save_area) ---
	ret;
}




.const .align 4 .u32 d_N;
.const .align 4 .u32 d_numRows;

.visible .entry _Z11matMulTiledPKfS0_Pf(
.param .u64 _Z11matMulTiledPKfS0_Pf_param_0,
.param .u64 _Z11matMulTiledPKfS0_Pf_param_1,
.param .u64 _Z11matMulTiledPKfS0_Pf_param_2
)
{
.reg .pred %p<6>;
.reg .f32 %f<105>;
.reg .b32 %r<35>;
.reg .b64 %rd<16>;
.shared .align 4 .b8 _ZZ11matMulTiledPKfS0_PfE2sA[4096];
.shared .align 4 .b8 _ZZ11matMulTiledPKfS0_PfE2sB[4096];

ld.param.u64 %rd5, [_Z11matMulTiledPKfS0_Pf_param_0];
call __memcapture__1;
ld.param.u64 %rd6, [_Z11matMulTiledPKfS0_Pf_param_1];
call __memcapture__2;
ld.param.u64 %rd7, [_Z11matMulTiledPKfS0_Pf_param_2];
call __memcapture__3;
mov.u32 %r18, %ctaid.y;
shl.b32 %r19, %r18, 5;
mov.u32 %r1, %tid.y;
add.s32 %r2, %r19, %r1;
mov.u32 %r20, %ctaid.x;
shl.b32 %r3, %r20, 5;
mov.u32 %r4, %tid.x;
add.s32 %r5, %r3, %r4;
ld.const.u32 %r6, [d_N];
call __memcapture__4;
setp.lt.s32 %p1, %r6, 32;
mov.f32 %f104, 0f00000000;
@%p1 bra $L__BB0_3;

shl.b32 %r22, %r1, 7;
mov.u32 %r23, _ZZ11matMulTiledPKfS0_PfE2sA;
add.s32 %r9, %r23, %r22;
shl.b32 %r24, %r4, 2;
add.s32 %r7, %r9, %r24;
mov.u32 %r25, _ZZ11matMulTiledPKfS0_PfE2sB;
add.s32 %r26, %r25, %r22;
add.s32 %r8, %r26, %r24;
add.s32 %r10, %r25, %r24;
mad.lo.s32 %r27, %r1, %r6, %r4;
add.s32 %r33, %r27, %r3;
shl.b32 %r12, %r6, 5;
mad.lo.s32 %r28, %r6, %r2, %r4;
cvta.to.global.u64 %rd8, %rd5;
mul.wide.s32 %rd9, %r28, 4;
add.s64 %rd15, %rd8, %rd9;
shr.s32 %r29, %r6, 31;
shr.u32 %r30, %r29, 27;
add.s32 %r31, %r6, %r30;
shr.s32 %r13, %r31, 5;
cvta.to.global.u64 %rd2, %rd6;
mov.f32 %f104, 0f00000000;
mov.u32 %r34, 0;

$L__BB0_2:
ld.global.f32 %f6, [%rd15];
st.shared.f32 [%r7], %f6;
mul.wide.s32 %rd10, %r33, 4;
add.s64 %rd11, %rd2, %rd10;
ld.global.f32 %f7, [%rd11];
st.shared.f32 [%r8], %f7;
bar.sync 0;
ld.shared.f32 %f8, [%r10];
ld.shared.f32 %f9, [%r9];
fma.rn.f32 %f10, %f9, %f8, %f104;
ld.shared.f32 %f11, [%r10+128];
ld.shared.f32 %f12, [%r9+4];
fma.rn.f32 %f13, %f12, %f11, %f10;
ld.shared.f32 %f14, [%r10+256];
ld.shared.f32 %f15, [%r9+8];
fma.rn.f32 %f16, %f15, %f14, %f13;
ld.shared.f32 %f17, [%r10+384];
ld.shared.f32 %f18, [%r9+12];
fma.rn.f32 %f19, %f18, %f17, %f16;
ld.shared.f32 %f20, [%r10+512];
ld.shared.f32 %f21, [%r9+16];
fma.rn.f32 %f22, %f21, %f20, %f19;
ld.shared.f32 %f23, [%r10+640];
ld.shared.f32 %f24, [%r9+20];
fma.rn.f32 %f25, %f24, %f23, %f22;
ld.shared.f32 %f26, [%r10+768];
ld.shared.f32 %f27, [%r9+24];
fma.rn.f32 %f28, %f27, %f26, %f25;
ld.shared.f32 %f29, [%r10+896];
ld.shared.f32 %f30, [%r9+28];
fma.rn.f32 %f31, %f30, %f29, %f28;
ld.shared.f32 %f32, [%r10+1024];
ld.shared.f32 %f33, [%r9+32];
fma.rn.f32 %f34, %f33, %f32, %f31;
ld.shared.f32 %f35, [%r10+1152];
ld.shared.f32 %f36, [%r9+36];
fma.rn.f32 %f37, %f36, %f35, %f34;
ld.shared.f32 %f38, [%r10+1280];
ld.shared.f32 %f39, [%r9+40];
fma.rn.f32 %f40, %f39, %f38, %f37;
ld.shared.f32 %f41, [%r10+1408];
ld.shared.f32 %f42, [%r9+44];
fma.rn.f32 %f43, %f42, %f41, %f40;
ld.shared.f32 %f44, [%r10+1536];
ld.shared.f32 %f45, [%r9+48];
fma.rn.f32 %f46, %f45, %f44, %f43;
ld.shared.f32 %f47, [%r10+1664];
ld.shared.f32 %f48, [%r9+52];
fma.rn.f32 %f49, %f48, %f47, %f46;
ld.shared.f32 %f50, [%r10+1792];
ld.shared.f32 %f51, [%r9+56];
fma.rn.f32 %f52, %f51, %f50, %f49;
ld.shared.f32 %f53, [%r10+1920];
ld.shared.f32 %f54, [%r9+60];
fma.rn.f32 %f55, %f54, %f53, %f52;
ld.shared.f32 %f56, [%r10+2048];
ld.shared.f32 %f57, [%r9+64];
fma.rn.f32 %f58, %f57, %f56, %f55;
ld.shared.f32 %f59, [%r10+2176];
ld.shared.f32 %f60, [%r9+68];
fma.rn.f32 %f61, %f60, %f59, %f58;
ld.shared.f32 %f62, [%r10+2304];
ld.shared.f32 %f63, [%r9+72];
fma.rn.f32 %f64, %f63, %f62, %f61;
ld.shared.f32 %f65, [%r10+2432];
ld.shared.f32 %f66, [%r9+76];
fma.rn.f32 %f67, %f66, %f65, %f64;
ld.shared.f32 %f68, [%r10+2560];
ld.shared.f32 %f69, [%r9+80];
fma.rn.f32 %f70, %f69, %f68, %f67;
ld.shared.f32 %f71, [%r10+2688];
ld.shared.f32 %f72, [%r9+84];
fma.rn.f32 %f73, %f72, %f71, %f70;
ld.shared.f32 %f74, [%r10+2816];
ld.shared.f32 %f75, [%r9+88];
fma.rn.f32 %f76, %f75, %f74, %f73;
ld.shared.f32 %f77, [%r10+2944];
ld.shared.f32 %f78, [%r9+92];
fma.rn.f32 %f79, %f78, %f77, %f76;
ld.shared.f32 %f80, [%r10+3072];
ld.shared.f32 %f81, [%r9+96];
fma.rn.f32 %f82, %f81, %f80, %f79;
ld.shared.f32 %f83, [%r10+3200];
ld.shared.f32 %f84, [%r9+100];
fma.rn.f32 %f85, %f84, %f83, %f82;
ld.shared.f32 %f86, [%r10+3328];
ld.shared.f32 %f87, [%r9+104];
fma.rn.f32 %f88, %f87, %f86, %f85;
ld.shared.f32 %f89, [%r10+3456];
ld.shared.f32 %f90, [%r9+108];
fma.rn.f32 %f91, %f90, %f89, %f88;
ld.shared.f32 %f92, [%r10+3584];
ld.shared.f32 %f93, [%r9+112];
fma.rn.f32 %f94, %f93, %f92, %f91;
ld.shared.f32 %f95, [%r10+3712];
ld.shared.f32 %f96, [%r9+116];
fma.rn.f32 %f97, %f96, %f95, %f94;
ld.shared.f32 %f98, [%r10+3840];
ld.shared.f32 %f99, [%r9+120];
fma.rn.f32 %f100, %f99, %f98, %f97;
ld.shared.f32 %f101, [%r10+3968];
ld.shared.f32 %f102, [%r9+124];
fma.rn.f32 %f104, %f102, %f101, %f100;
bar.sync 0;
add.s32 %r33, %r33, %r12;
add.s64 %rd15, %rd15, 128;
add.s32 %r34, %r34, 1;
setp.lt.s32 %p2, %r34, %r13;
@%p2 bra $L__BB0_2;

$L__BB0_3:
setp.ge.s32 %p3, %r5, %r6;
setp.ge.s32 %p4, %r2, %r6;
or.pred %p5, %p4, %p3;
@%p5 bra $L__BB0_5;

mad.lo.s32 %r32, %r6, %r2, %r5;
cvta.to.global.u64 %rd12, %rd7;
mul.wide.s32 %rd13, %r32, 4;
add.s64 %rd14, %rd12, %rd13;
st.global.f32 [%rd14], %f104;

$L__BB0_5:
ret;

}


.visible .entry kernel_2(
.param .u64 _Z11matMulTiledPKfS0_Pf_param_0,
.param .u64 _Z11matMulTiledPKfS0_Pf_param_1,
.param .u64 _Z11matMulTiledPKfS0_Pf_param_2
)
{
.reg .pred %p<6>;
.reg .f32 %f<105>;
.reg .b32 %r<35>;
.reg .b64 %rd<16>;
{}{}{{}{{}{}}}
ret;

}


)";

TEST_CASE("Test kernel finder")
{
	bpftime::attach::nv_attach_impl impl;
	impl.patch_with_probe_and_retprobe(
		ORIGINAL_PTX,
		bpftime::attach::nv_attach_entry{
			.type =
				bpftime::attach::nv_attach_function_probe{
					.func = "kernel_2",
					.is_retprobe = false,
				},
			.instuctions = std::vector<ebpf_inst>(),

		},
		true);
	std::string str = ORIGINAL_PTX;
	std::cout << str.substr(41242, 41502 - 41242);
	REQUIRE(false);
}
