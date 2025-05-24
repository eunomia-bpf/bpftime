#include <__clang_cuda_builtin_vars.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

/* clang++-17 -S ./default_trampoline.cu -Wall --cuda-gpu-arch=sm_60 -O2 -L/usr/local/cuda/lib64/ -lcudart*/
enum class HelperOperation {
	MAP_LOOKUP = 1,
	MAP_UPDATE = 2,
	MAP_DELETE = 3,
	MAP_GET_NEXT_KEY = 4,
	TRACE_PRINTK = 6,
	GET_CURRENT_PID_TGID = 14,
	PUTS = 501
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
	struct {
		char data[10000];
	} puts;
	struct {
	} get_tid_pgid;
};

union HelperCallResponse {
	struct {
		int result;
	} map_update, map_delete, trace_printk, puts;
	struct {
		const void *value;
	} map_lookup;
	struct {
		uint64_t result;
	} get_tid_pgid;
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

__device__ __forceinline__ uint64_t read_globaltimer()
{
	uint64_t timestamp;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
	return timestamp;
}

__constant__ uintptr_t constData;
__constant__ MapBasicInfo map_info[256];
extern "C" __device__ void spin_lock(volatile int *lock)
{
	while (atomicCAS((int *)lock, 0, 1) == 1) {
		// 自旋等待锁变为可用
	}
	// printf("lock acquired by %d\n", threadIdx.x + blockIdx.x *
	// blockDim.x);
}

extern "C" __device__ void spin_unlock(int *lock)
{
	atomicExch(lock, 0); // 将锁标志重置为 0
	// printf("lock released by %d\n", threadIdx.x + blockIdx.x *
	// blockDim.x);
}
extern "C" __device__ HelperCallResponse make_helper_call(long map_id,
							  int req_id)
{
	CommSharedMem *g_data = (CommSharedMem *)constData;
	// printf("make_map_call at %d, constdata=%lx\n",
	//        threadIdx.x + blockIdx.x * blockDim.x, (uintptr_t)g_data);
	// auto start_time = read_globaltimer();
	spin_lock(&g_data->occupy_flag);
	// 准备要写入的参数值
	int val = 42; // 这里就写一个固定值，示例用
	// g_data->req = req;
	g_data->request_id = req_id;
	g_data->map_id = map_id;
	// printf("making call for %d\n", req_id);
	// 在内联PTX里演示 store/load + acquire/release + 自旋
	asm volatile(
		".reg .pred p0;                   \n\t" // 声明谓词寄存器
		"membar.sys;                      \n\t" // 内存屏障
							// 设置 flag1 = 1 (替代
							// st.global.rel.u32)
		"st.global.u32 [%1], 1;           \n\t"
		// 自旋等待 flag2 == 1 (替代 ld.global.acq.u32)
		"spin_wait:                       \n\t"
		"membar.sys;                      \n\t"
		"ld.global.u32 %0, [%2];          \n\t" // 读取 flag2
		"setp.eq.u32 p0, %0, 0;           \n\t" // 比较值
		"@p0 bra spin_wait;               \n\t" // 谓词分支
							// 若跳出循环，复位
							// flag2 = 0
		"st.global.u32 [%2], 0;           \n\t"
		"membar.sys;                      \n\t"
		:
		: "r"(val), "l"(&g_data->flag1), "l"(&g_data->flag2)
		: "memory");
	HelperCallResponse resp = g_data->resp;

	spin_unlock(&g_data->occupy_flag);
	// auto end_time = read_globaltimer();
	// if (req_id < 8) {
	// 	atomicAdd((unsigned long long *)&g_data->time_sum[req_id],
	// 		  end_time - start_time);
	// }
	return resp;
}

extern "C" __device__ inline void simple_memcpy(void *dst, void *src, int sz)
{
	for (int i = 0; i < sz; i++)
		((char *)dst)[i] = ((char *)src)[i];
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0001(
	uint64_t map, uint64_t key, uint64_t a, uint64_t b, uint64_t c)
{
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req;
	// CallRequest req;
	const auto &map_info = ::map_info[map];
	// printf("helper1 map %ld keysize=%d valuesize=%d\n", map,
	//        map_info.key_size, map_info.value_size);
	simple_memcpy(&req.map_lookup.key, (void *)(uintptr_t)key,
		      map_info.key_size);

	HelperCallResponse resp =
		make_helper_call((long)map, (int)HelperOperation::MAP_LOOKUP);

	return (uintptr_t)resp.map_lookup.value;
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0002(
	uint64_t map, uint64_t key, uint64_t value, uint64_t flags, uint64_t a)
{
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req;
	const auto &map_info = ::map_info[map];
	// printf("helper2 map %ld keysize=%d
	// valuesize=%d\n",map,map_info.key_size,map_info.value_size);
	simple_memcpy(&req.map_update.key, (void *)(uintptr_t)key,
		      map_info.key_size);
	simple_memcpy(&req.map_update.value, (void *)(uintptr_t)value,
		      map_info.value_size);
	req.map_update.flags = (uintptr_t)flags;

	HelperCallResponse resp =
		make_helper_call((long)map, (int)HelperOperation::MAP_UPDATE);
	return resp.map_update.result;
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0003(
	uint64_t map, uint64_t key, uint64_t a, uint64_t b, uint64_t c)
{
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req;
	const auto &map_info = ::map_info[map];
	// printf("helper3 map %ld keysize=%d
	// valuesize=%d\n",map,map_info.key_size,map_info.value_size);
	simple_memcpy(&req.map_delete.key, (void *)(uintptr_t)key,
		      map_info.key_size);
	HelperCallResponse resp =
		make_helper_call((long)map, (int)HelperOperation::MAP_DELETE);
	return resp.map_delete.result;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0006(uint64_t fmt, uint64_t fmt_size, uint64_t arg1,
		     uint64_t arg2, uint64_t arg3)
{
	// printf("Calling 0006 fmt %s\n",(char*)fmt);
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req;
	char *out = (char *)req.trace_printk.fmt;
	char *in = (char *)(uintptr_t)fmt;
	for (auto i = 0; i < fmt_size; i++) {
		out[i] = in[i];
	}
	req.trace_printk.fmt_size = fmt_size;
	req.trace_printk.arg1 = arg1;
	req.trace_printk.arg2 = arg2;
	req.trace_printk.arg3 = arg3;
	HelperCallResponse resp =
		make_helper_call(0, (int)HelperOperation::TRACE_PRINTK);
	return resp.trace_printk.result;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0014(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	HelperCallResponse resp =
		make_helper_call(0, (int)HelperOperation::GET_CURRENT_PID_TGID);
	return resp.get_tid_pgid.result;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0501(uint64_t data, uint64_t, uint64_t, uint64_t, uint64_t)
{
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req.puts;

	const char *input = (const char *)data;
	int idx = 0;
	while (input[idx]) {
		req.data[idx] = input[idx];
		idx++;
	}
	req.data[idx] = 0;
	HelperCallResponse resp =
		make_helper_call(0, (int)HelperOperation::PUTS);
	return resp.puts.result;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0502(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	return read_globaltimer();
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0503(uint64_t x, uint64_t y, uint64_t z, uint64_t, uint64_t)
{
	// get block idx
	*(uint64_t *)(uintptr_t)x = blockIdx.x;
	*(uint64_t *)(uintptr_t)y = blockIdx.y;
	*(uint64_t *)(uintptr_t)z = blockIdx.z;

	return 0;
}
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0504(uint64_t x, uint64_t y, uint64_t z, uint64_t, uint64_t)
{
	// get block dim
	*(uint64_t *)(uintptr_t)x = blockDim.x;
	*(uint64_t *)(uintptr_t)y = blockDim.y;
	*(uint64_t *)(uintptr_t)z = blockDim.z;

	return 0;
}
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0505(uint64_t x, uint64_t y, uint64_t z, uint64_t, uint64_t)
{
	// get threadIdx
	*(uint64_t *)(uintptr_t)x = threadIdx.x;
	*(uint64_t *)(uintptr_t)y = threadIdx.y;
	*(uint64_t *)(uintptr_t)z = threadIdx.z;

	return 0;
}

extern "C" __global__ void bpf_main(void *mem, size_t sz)
{
	printf("kernel function entered, mem=%lx, memsz=%ld\n", (uintptr_t)mem,
	       sz);
	char buf[16] = "aaa";
	printf("setup function, const data=%lx\n", constData);
	auto result = _bpf_helper_ext_0001(1ull << 32, (uintptr_t)buf, 0, 0, 0);
	_bpf_helper_ext_0002(1ull << 32, (uintptr_t)buf, (uintptr_t)buf, 0, 0);
	_bpf_helper_ext_0003(1ull << 32, (uintptr_t)buf, 0, 0, 0);
	const char msg[] = "Message from bpf: %d, %lx";
	_bpf_helper_ext_0006((uint64_t)(uintptr_t)msg, sizeof(msg), 10, 20, 0);

	printf("call done\n");
	printf("got response %d at %d\n", *(int *)result,
	       threadIdx.x + blockIdx.x * blockDim.x);
	*(int *)mem = 123;
}

static std::atomic<bool> should_exit;
void signal_handler(int)
{
	should_exit.store(true);
}
int main()
{
	signal(SIGINT, signal_handler);

	// 1. 先在主机上分配一块普通内存
	CommSharedMem *hostMem = (CommSharedMem *)malloc(sizeof(CommSharedMem));
	if (!hostMem) {
		std::cerr << "Failed to allocate hostMem\n";
		return -1;
	}

	// 2. 注册成 pinned memory (可被GPU直接访问)
	cudaError_t err = cudaHostRegister(hostMem, sizeof(CommSharedMem),
					   cudaHostRegisterMapped);
	if (err != cudaSuccess) {
		std::cerr
			<< "cudaHostRegister error: " << cudaGetErrorString(err)
			<< "\n";
		free(hostMem);
		return -1;
	}

	// 3. 获取对应的设备指针(这样DeviceKernel就能直接访问这个地址)
	CommSharedMem *devPtr = nullptr;
	err = cudaHostGetDevicePointer((void **)&devPtr, (void *)hostMem, 0);
	if (err != cudaSuccess) {
		std::cerr << "cudaHostGetDevicePointer error: "
			  << cudaGetErrorString(err) << "\n";
		cudaHostUnregister(hostMem);
		free(hostMem);
		return -1;
	}
	printf("dev ptr should be %lx, host ptr is %lx\n", (uintptr_t)devPtr,
	       (uintptr_t)hostMem);
	err = cudaMemcpyToSymbol(constData, &devPtr, sizeof(CommSharedMem *));
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpyToSymbol error: "
			  << cudaGetErrorString(err) << "\n";
		cudaHostUnregister(hostMem);
		free(hostMem);
		return -1;
	}
	int buf = 11223344;
	err = cudaHostRegister((void *)&buf, sizeof(buf),
			       cudaHostRegisterMapped);
	if (err != cudaSuccess) {
		std::cerr << "cudaHostRegister(2) error: "
			  << cudaGetErrorString(err) << " " << err << "\n";
		cudaHostUnregister(hostMem);
		free(hostMem);
		return -1;
	}
	char *devPtrStr = nullptr;
	err = cudaHostGetDevicePointer((void **)&devPtrStr, (void *)&buf, 0);
	if (err != cudaSuccess) {
		std::cerr << "cudaHostGetDevicePointer(2) error: "
			  << cudaGetErrorString(err) << "\n";
		cudaHostUnregister(hostMem);
		free(hostMem);
		return -1;
	}
	// 初始化标志位
	memset(hostMem, 0, sizeof(*hostMem));
	// 4. 启动一个线程, 模拟host侧的处理逻辑
	std::thread hostThread([&]() {
		std::cout << "[Host Thread] Start waiting...\n";

		// 这里简单用轮询，检测到flag1=1就处理
		while (!should_exit.load()) {
			if (hostMem->flag1 == 1) {
				// 清掉flag1防止重复处理
				hostMem->flag1 = 0;
				// 假设处理数据 paramA
				std::cout
					<< "[Host Thread] Got request: req_id="
					<< hostMem->request_id
					<< ", handling...\n";
				if (hostMem->request_id == 1) {
					std::cout << "call map_lookup="
						  << hostMem->req.map_lookup.key
						  << std::endl;
					// strcpy(hostMem->resp.map_lookup.value,
					//        "your value");
					hostMem->resp.map_lookup.value =
						devPtrStr;
				}
				// std::atomic_thread_fence(std::memory_order_seq_cst);

				// 处理完后, 把 flag2=1, 让设备端退出自旋
				hostMem->flag2 = 1;

				// 在实际开发中，可以加个内存栅栏，比如：
				std::atomic_thread_fence(
					std::memory_order_seq_cst);

				// 处理一次就退出本线程循环
				// break;
				std::cout << "handle done, timesum = "
					  << hostMem->time_sum[1] << std::endl;
			}

			// 为了演示，这里短暂休眠，避免100%占用CPU
			std::this_thread::sleep_for(
				std::chrono::milliseconds(10));
		}

		std::cout << "[Host Thread] Done.\n";
	});
	std::vector<MapBasicInfo> local_map_info(256);

	local_map_info[1].enabled = true;
	local_map_info[1].key_size = 16;
	local_map_info[1].value_size = 16;
	cudaMemcpyToSymbol(map_info, local_map_info.data(),
			   sizeof(MapBasicInfo) * local_map_info.size());
	// 5. 启动核函数 (只发1个block,1个thread做演示)
	bpf_main<<<1, 1>>>(hostMem, sizeof(*hostMem));

	// 等待核函数执行完毕
	cudaDeviceSynchronize();

	// 等待host线程结束
	hostThread.join();

	// 6. 收尾：解绑 pinned memory 并释放
	cudaHostUnregister(hostMem);
	free(hostMem);

	std::cout << "All done.\n";
	return 0;
}
