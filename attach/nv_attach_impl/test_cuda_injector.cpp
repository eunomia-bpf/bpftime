#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>
#include <regex>
#include "nv_attach_impl.hpp"

TEST_CASE("Test CUDAInjector - basic attach/detach")
{
	// For demonstration, pick a dummy or real PID.
	// In a real-world test, you'd spawn a child process running a CUDA app.
	pid_t test_pid = 2663926;

	// 1. Construct the injector
	bpftime::attach::CUDAInjector injector(test_pid, "../example/cudamem-capture/victim.ptx");

	// 2. Attempt to attach to the process
	bool attached = injector.attach();
	REQUIRE(attached == true);

	// 3. [Optional] Attempt to inject PTX code
	SECTION("Inject PTX code")
	{
		// A trivial PTX kernel as an example
		const char *ptx_code = R"(.entry probe_infinite_kernel__cuda() {
    // A do-nothing kernel
    ret;
})";

		// Suppose we want to inject at some device memory address
		// (dummy).
		CUdeviceptr dummy_inject_addr = 0x10000000;
		size_t dummy_code_size = 256; // Example

		// A hypothetical method in CUDAInjector for demonstration
		bool success = injector.inject_ptx(ptx_code, dummy_inject_addr,
						   dummy_code_size);
		REQUIRE(success == true);
	}

	// 4. Detach from the process
	bool detached = injector.detach();
	REQUIRE(detached == true);

	// 5. Attempting to attach again or inject code after detaching might
	// fail
	//    but you can add negative tests if you want:
	SECTION("Attach again (negative test)")
	{
		bool reattach = injector.attach();
		REQUIRE(reattach == true);
	}
}

TEST_CASE("Test String Concat - ptx load/unload")
{
	// 原始PTX代码
	std::string ptx_code = R"(//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-33191640
// Cuda compilation tools, release 12.2, V12.2.140
// Based on NVVM 7.0.1
//
.version 8.2
.target sm_70
.address_size 64
	// .globl	infinite_kernel
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.global .align 4 .u32 should_exit;
.global .align 1 .b8 $str[16] = {75, 101, 114, 110, 101, 108, 32, 115, 116, 97, 114, 116, 101, 100, 10};
.global .align 1 .b8 $str$1[29] = {83, 116, 105, 108, 108, 32, 114, 117, 110, 110, 105, 110, 103, 46, 46, 46, 32, 99, 111, 117, 110, 116, 101, 114, 61, 37, 100, 10};
.global .align 4 .b8 __cudart_i2opi_f[24] = {65, 144, 67, 60, 153, 149, 98, 219, 192, 221, 52, 245, 209, 87, 39, 252, 41, 21, 68, 78, 110, 131, 249, 162};
.visible .entry infinite_kernel()
{
	.local .align 8 .b8 	__local_depot0[40];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<16>;
	.reg .f32 	%f<41>;
	.reg .b32 	%r<79>;
	.reg .f64 	%fd<3>;
	.reg .b64 	%rd<29>;
})";

	// probe函数定义
	std::string probe_func = R"(.entry probe_infinite_kernel__cuda() {
    // A do-nothing kernel
    abcd;
    ret;
})";

	// 使用正则表达式匹配probe函数
	std::regex probe_regex(
		R"(.entry probe_(.+)__cuda\(.*\) \{((.*\n)*)((.*ret;\s*\n)*)\})");
	std::smatch probe_match;

	if (std::regex_search(probe_func, probe_match, probe_regex)) {
		// 从匹配结果中提取函数名和函数体
		std::string function_body = probe_match[2];
		std::string function_name_part =
			probe_match[1]; // 获取函数名部分
		std::cout << "函数体: " << function_body
			  << "\n函数名部分: " << function_name_part
			  << std::endl;

		// 从函数体中移除ret指令
		std::string body_without_ret = function_body;
		size_t ret_pos = body_without_ret.find("ret;");
		if (ret_pos != std::string::npos) {
			// 找到ret;所在行的开始位置
			size_t line_start =
				body_without_ret.rfind('\n', ret_pos);
			if (line_start == std::string::npos)
				line_start = 0;
			else
				line_start++; // 跳过换行符

			// 找到ret;所在行的结束位置
			size_t line_end = body_without_ret.find('\n', ret_pos);
			if (line_end == std::string::npos)
				line_end = body_without_ret.length();
			else
				line_end++; // 包含换行符

			// 移除整行
			body_without_ret.erase(line_start,
					       line_end - line_start);
		}

		// 找到最后一个.reg声明的位置
		std::regex reg_pattern(R"(\.reg \.b64 \t%rd<\d+>;)");
		std::smatch reg_match;
		std::string modified_ptx = ptx_code;

		if (std::regex_search(ptx_code, reg_match, reg_pattern)) {
			// 找到匹配的位置
			size_t insert_pos =
				reg_match.position() + reg_match.length();

			// 在.reg声明后插入probe函数体
			modified_ptx.insert(insert_pos,
					    "\n" + body_without_ret);
		}

		// 输出修改后的代码
		std::cout << "修改后的代码：\n" << modified_ptx << std::endl;
	} else {
		std::cout << "正则表达式未能匹配probe函数" << std::endl;
	}
	// retprobe函数定义
	std::string probe_func1 = R"(.entry retprobe_infinite_kernel__cuda() {
    // A do-nothing kernel
    abcd;
    ret;
})";

	// 使用正则表达式匹配retprobe函数
	std::regex probe_regex1(
		R"(.entry (retprobe)_(.+)__cuda\(.*\) \{((.*\n)*)((.*ret;\s*\n)*)\})");
	std::smatch probe_match1;

	if (std::regex_search(probe_func1, probe_match1, probe_regex1)) {
		// 从匹配结果中提取函数名和函数体
		std::string probe_name = probe_match1[1]; // retprobe
		std::string function_name = probe_match1[2]; // infinite_kernel
		std::string function_body = probe_match1[3]; // 函数体

		std::cout << "探针名: " << probe_name << "\n"
			  << "函数名: " << function_name << "\n"
			  << "函数体: " << function_body << std::endl;

		// 从函数体中移除ret指令
		std::string body_without_ret = function_body;
		size_t ret_pos = body_without_ret.find("ret;");
		if (ret_pos != std::string::npos) {
			// 找到ret;所在行的开始位置
			size_t line_start =
				body_without_ret.rfind('\n', ret_pos);
			if (line_start == std::string::npos)
				line_start = 0;
			else
				line_start++; // 跳过换行符

			// 找到ret;所在行的结束位置
			size_t line_end = body_without_ret.find('\n', ret_pos);
			if (line_end == std::string::npos)
				line_end = body_without_ret.length();
			else
				line_end++; // 包含换行符

			// 移除整行
			body_without_ret.erase(line_start,
					       line_end - line_start);
		}

		// 找到目标函数的右花括号位置
		std::regex target_func_regex(R"(\.visible \.entry )" +
					     function_name +
					     R"(\(\)[\s\S]*?\})");
		std::smatch target_match;

		if (std::regex_search(ptx_code, target_match,
				      target_func_regex)) {
			std::string target_func = target_match[0];
			size_t closing_brace_pos = target_match.position() + target_func.rfind("}");

            // 在右花括号前插入retprobe函数体
            std::string modified_ptx = ptx_code;
            modified_ptx.insert(closing_brace_pos, "\n" + body_without_ret);

            // 输出修改后的代码
            std::cout << "修改后的代码：\n" << modified_ptx << std::endl;
        } else {
            std::cout << "未能找到目标函数 " << function_name << std::endl;
        }
    } else {
        std::cout << "正则表达式未能匹配retprobe函数" << std::endl;
    }

}
