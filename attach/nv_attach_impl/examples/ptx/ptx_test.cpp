#include <cassert>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm_jit_context.hpp>
#include <ebpf_inst.h>
#include <ostream>

#include <cuda.h>
#include <nvPTXCompiler.h>
#include <sstream>
#include <trampoline_ptx.h>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <nv_attach_impl.hpp>
using namespace bpftime;
using namespace std;

static llvm::ExitOnError exitOnError;

static uint64_t test_func(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	return 0;
}

#define CUDA_SAFE_CALL(x)                                                      \
	do {                                                                   \
		CUresult result = x;                                           \
		if (result != CUDA_SUCCESS) {                                  \
			const char *msg;                                       \
			cuGetErrorName(result, &msg);                          \
			printf("error: %s failed with error %s\n", #x, msg);   \
			exit(1);                                               \
		}                                                              \
	} while (0)

#define CUDA_SAFE_CALL_2(x)                                                    \
	do {                                                                   \
		cudaError_t result = x;                                        \
		if (result != cudaSuccess) {                                   \
			printf("error: %s failed with error %s\n", #x,         \
			       cudaGetErrorString(result));                    \
			exit(1);                                               \
		}                                                              \
	} while (0)

#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
	do {                                                                   \
		nvPTXCompileResult result = x;                                 \
		if (result != NVPTXCOMPILE_SUCCESS) {                          \
			printf("error: %s failed with error code %d\n", #x,    \
			       result);                                        \
			exit(1);                                               \
		}                                                              \
	} while (0)
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
	// r1 = 1<<32
	{ EBPF_OP_LDDW, 1, 0, 0, 0 },
	{ 0, 0, 0, 0, 1 },

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

static std::vector<char> compile(const std::string &ptx)
{
	nvPTXCompilerHandle compiler = NULL;
	nvPTXCompileResult status;

	size_t elfSize, infoSize, errorSize;
	unsigned int minorVer, majorVer;

	const char *compile_options[] = { "--gpu-name=sm_60", "--verbose" };

	NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
	printf("Current PTX Compiler API Version : %d.%d\n", majorVer,
	       minorVer);

	NVPTXCOMPILER_SAFE_CALL(
		nvPTXCompilerCreate(&compiler, (size_t)ptx.size(), /* ptxCodeLen
								    */
				    ptx.c_str()) /* ptxCode */
	);

	status = nvPTXCompilerCompile(compiler, 2, /* numCompileOptions */
				      compile_options); /* compileOptions */

	if (status != NVPTXCOMPILE_SUCCESS) {
		NVPTXCOMPILER_SAFE_CALL(
			nvPTXCompilerGetErrorLogSize(compiler, &errorSize));
		std::string error(errorSize, 0);
		if (errorSize != 0) {
			NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(
				compiler, (char *)error.c_str()));
			printf("Error log: %s\n", error.c_str());
		}
		exit(1);
	}

	NVPTXCOMPILER_SAFE_CALL(
		nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
	std::vector<char> elf_binary(elfSize, 0);
	NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
		compiler, (void *)elf_binary.data()));

	NVPTXCOMPILER_SAFE_CALL(
		nvPTXCompilerGetInfoLogSize(compiler, &infoSize));
	std::string info(infoSize, 0);
	if (infoSize != 0) {
		NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(
			compiler, (char *)info.c_str()));
		printf("Info log: %s\n", info.c_str());
	}
	NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
	return elf_binary;
}

enum class HelperOperation {
	MAP_LOOKUP = 1,
	MAP_UPDATE = 2,
	MAP_DELETE = 3,
	MAP_GET_NEXT_KEY = 4,
	TRACE_PRINTK = 6
};

union HelperCallRequest {
	struct {
		char key[1 << 30];
	} map_lookup;
	struct {
		char key[1 << 30];
		char value[1 << 30];
		uint64_t flags;
	} map_update;
	struct {
		char key[1 << 30];
	} map_delete;
	struct {
		char fmt[1000];
		int fmt_size;
		unsigned long arg1, arg2, arg3;
	} trace_printk;
};

union HelperCallResponse {
	struct {
		int result;
	} map_update, map_delete, trace_printk;
	struct {
		const void *value;
	} map_lookup;
};
/**
 * 我们在这块结构体里放两个标志位和一个简单的参数字段
 * - flag1: device -> host 的信号，“我有请求要处理”
 * - flag2: host   -> device 的信号，“我处理完了”
 * - paramA: 设备端写入的参数，让主机端使用
 */
struct CommSharedMem {
	int flag1;
	int flag2;
	int occupy_flag;
	int request_id;
	long map_id;
	HelperCallRequest req;
	HelperCallResponse resp;
	uint64_t time_sum[8];
};
struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
};
static std::atomic<bool> should_exit;
void signal_handler(int)
{
	should_exit.store(true);
}

static int elfLoadAndKernelLaunch(void *elf, size_t elfSize)
{
	CUdevice cuDevice;
	CUcontext context;
	CUmodule module;
	CUfunction kernel;

	CUDA_SAFE_CALL(cuInit(0));
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

	CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
	CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
	auto comm = std::make_unique<CommSharedMem>();
	memset(comm.get(), 0, sizeof(CommSharedMem));
	{
		CUdeviceptr constDataPtr;
		size_t constDataLen;

		CUDA_SAFE_CALL(cuModuleGetGlobal(&constDataPtr, &constDataLen,
						 module, "constData"));
		cout << "const data length=" << constDataLen << endl;
		CUDA_SAFE_CALL(cuMemHostRegister(comm.get(),
						 sizeof(CommSharedMem),
						 CU_MEMHOSTREGISTER_DEVICEMAP));
		CUdeviceptr memDevPtr;
		CUDA_SAFE_CALL(
			cuMemHostGetDevicePointer(&memDevPtr, comm.get(), 0));
		CUDA_SAFE_CALL(cuMemcpyHtoD(constDataPtr, &memDevPtr,
					    sizeof(memDevPtr)));
	}
	{
		CUdeviceptr map_info;
		size_t map_info_len;
		CUDA_SAFE_CALL(cuModuleGetGlobal(&map_info, &map_info_len,
						 module, "map_info"));

		std::vector<MapBasicInfo> local_map_info(256);

		local_map_info[1].enabled = true;
		local_map_info[1].key_size = 16;
		local_map_info[1].value_size = 16;
		CUDA_SAFE_CALL(cuMemcpyHtoD(map_info, local_map_info.data(),
					    sizeof(MapBasicInfo) *
						    local_map_info.size()));
	}
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "bpf_main"));

	uint64_t mem_size = 1024;
	std::vector<char> sharedMem(mem_size, 0);
	CUDA_SAFE_CALL(cuMemHostRegister(sharedMem.data(), sharedMem.size(),
					 CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr memDevPtr;
	CUDA_SAFE_CALL(
		cuMemHostGetDevicePointer(&memDevPtr, sharedMem.data(), 0));
	void *args[2] = { &memDevPtr, &mem_size };

	CUDA_SAFE_CALL(cuLaunchKernel(kernel, 1, 1, 1, // grid dim
				      1, 1, 1, // block dim
				      0, nullptr, // shared mem and stream
				      args, 0)); // arguments
	auto response = std::make_unique<int>(11223344);
	CUDA_SAFE_CALL(cuMemHostRegister(response.get(), sizeof(int),
					 CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr respDevPtr;
	CUDA_SAFE_CALL(
		cuMemHostGetDevicePointer(&respDevPtr, response.get(), 0));
	std::thread hostThread([&]() {
		std::cout << "[Host Thread] Start waiting...\n";

		// 这里简单用轮询，检测到flag1=1就处理
		while (!should_exit.load()) {
			if (comm->flag1 == 1) {
				// 清掉flag1防止重复处理
				comm->flag1 = 0;
				// 假设处理数据 paramA
				std::cout
					<< "[Host Thread] Got request: req_id="
					<< comm->request_id
					<< ", handling...\n";
				if (comm->request_id == 1) {
					std::cout << "call map_lookup="
						  << comm->req.map_lookup.key
						  << std::endl;
					// strcpy(hostMem->resp.map_lookup.value,
					//        "your value");
					comm->resp.map_lookup.value =
						(const void *)respDevPtr;
				}
				// std::atomic_thread_fence(std::memory_order_seq_cst);

				// 处理完后, 把 flag2=1, 让设备端退出自旋
				comm->flag2 = 1;

				// 在实际开发中，可以加个内存栅栏，比如：
				std::atomic_thread_fence(
					std::memory_order_seq_cst);

				// 处理一次就退出本线程循环
				// break;
				std::cout << "handle done" << std::endl;
			}

			// 为了演示，这里短暂休眠，避免100%占用CPU
			std::this_thread::sleep_for(
				std::chrono::milliseconds(10));
		}

		std::cout << "[Host Thread] Done.\n";
	});
	CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.
	hostThread.join();
	cout << "first 4byte of memory = " << *(int *)sharedMem.data() << endl;
	CUDA_SAFE_CALL(cuMemHostUnregister(comm.get()));
	CUDA_SAFE_CALL(cuMemHostUnregister(response.get()));
	CUDA_SAFE_CALL(cuMemHostUnregister(sharedMem.data()));

	CUDA_SAFE_CALL(cuModuleUnload(module));
	CUDA_SAFE_CALL(cuCtxDestroy(context));
	return 0;
}

static std::string load_local_ptx()
{
	std::ifstream ifs("../test.ptx");

	return std::string((std::istreambuf_iterator<char>(ifs)),
			   std::istreambuf_iterator<char>());
}
int main()
{
	signal(SIGINT, signal_handler);
	llvm::InitializeAllTargetInfos(); // 初始化 TargetInfo
	llvm::InitializeAllTargets(); // 初始化 Target (注册 Target 对象)
	llvm::InitializeAllTargetMCs(); // 初始化 TargetMachine 创建所需内容
	llvm::InitializeAllAsmPrinters(); // 初始化汇编打印器
	llvm::InitializeAllAsmParsers(); // 如果需要解析汇编或 .ll 文件
	for (const auto &target : llvm::TargetRegistry::targets()) {
		cout << "Registered target: " << target.getName() << endl;
	}

	llvmbpf_vm vm;
	vm.register_external_function(1, "map_lookup", (void *)test_func);
	vm.register_external_function(2, "map_update", (void *)test_func);
	vm.register_external_function(3, "map_delete", (void *)test_func);

	vm.load_code((void *)test_prog, sizeof(test_prog));
	llvm_bpf_jit_context ctx(vm);
	auto result = *ctx.generate_ptx(false, "bpf_main", "sm_60");
	{
		std::ofstream ofs_result("out.ptx");
		ofs_result << result;
	}
	result = bpftime::attach::wrap_ptx_with_trampoline(bpftime::attach::patch_helper_names_and_header(
		bpftime::attach::patch_main_from_func_to_entry(result)));
	// auto result = load_local_ptx();
	cout << result << std::endl;
	auto bin = compile(result);
	// std::ofstream ofs("out.bin", ios::binary);
	// ofs.write(bin.data(), bin.size());
	// ofs.flush();
	elfLoadAndKernelLaunch(bin.data(), bin.size());
	return 0;
}
