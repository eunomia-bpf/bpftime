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
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

/* clang++-17 -S ./default_trampoline.cu -Wall --cuda-gpu-arch=sm_60 -O2
 * -L/usr/local/cuda/lib64/ -lcudart*/
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

const int BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP = 1502;
const int BPF_MAP_TYPE_GPU_ARRAY_MAP = 1503; // non-per-thread, single-copy shared array
const int BPF_MAP_TYPE_GPU_RINGBUF_MAP = 1527;

struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
	int map_type;
	void *extra_buffer;
	uint64_t max_thread_count;
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
	int lane_id = threadIdx.x & 31;
	HelperCallResponse my_resp = {};

	for (int active_lane = 0; active_lane < 32; active_lane++) {
		unsigned int active_mask = __activemask();
		bool lane_is_active = (active_mask >> active_lane) & 1;

		if (lane_is_active && lane_id == active_lane) {
			spin_lock(&g_data->occupy_flag);

			int val = 42;
			g_data->request_id = req_id;
			g_data->map_id = map_id;

			asm volatile(".reg .pred p0;                   \n\t"
				     "membar.sys;                      \n\t"
				     "st.global.u32 [%1], 1;           \n\t"
				     "spin_wait:                       \n\t"
				     "membar.sys;                      \n\t"
				     "ld.global.u32 %0, [%2];          \n\t"
				     "setp.eq.u32 p0, %0, 0;           \n\t"
				     "@p0 bra spin_wait;               \n\t"
				     "st.global.u32 [%2], 0;           \n\t"
				     "membar.sys;                      \n\t"
				     :
				     : "r"(val), "l"(&g_data->flag1),
				       "l"(&g_data->flag2)
				     : "memory");

			my_resp = g_data->resp;

			spin_unlock(&g_data->occupy_flag);
		}

		__syncwarp(active_mask);
	}

	return my_resp;
}
extern "C" __device__ inline void simple_memcpy(void *dst, void *src, int sz)
{
	for (int i = 0; i < sz; i++)
		((char *)dst)[i] = ((char *)src)[i];
}

__device__ uint64_t getGlobalThreadId()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.y;
	return ((uint64_t)z * width * height) + (y * width) + x;
}
__device__ void *array_map_offset(uint64_t idx, const MapBasicInfo &info)
{
	return (void *)((uintptr_t)info.extra_buffer +
			idx * info.max_thread_count * info.value_size +
			getGlobalThreadId() * info.value_size);
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0001(
	uint64_t map, uint64_t key, uint64_t a, uint64_t b, uint64_t c)
{
	CommSharedMem *global_data = (CommSharedMem *)constData;
	auto &req = global_data->req;
	// CallRequest req;
	const auto &map_info = ::map_info[map];
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info);
		return (uint64_t)offset;
	}
	// Fast-path for non-per-thread GPU array map: single shared copy on device-visible UVA
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		return (uint64_t)(uintptr_t)(base + (uint64_t)real_key * map_info.value_size);
	}
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
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info);
		simple_memcpy(offset, (void *)(uintptr_t)value,
			      map_info.value_size);
		return 0;
	}
	// Fast-path for non-per-thread GPU array map: memcpy overwrite, system fence for visibility
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		auto dst = (void *)(uintptr_t)(base + (uint64_t)real_key * map_info.value_size);
		simple_memcpy(dst, (void *)(uintptr_t)value, map_info.value_size);
		asm("membar.sys;                      \n\t");
		return 0;
	}
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
struct ringbuf_header {
	uint64_t head;
	uint64_t tail;
	int dirty;
};

// perf event output
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0025(uint64_t ctx, uint64_t map, uint64_t flags, uint64_t data,
		     uint64_t data_size)
{
	const auto &map_info = ::map_info[map];
	if (map_info.map_type == BPF_MAP_TYPE_GPU_RINGBUF_MAP) {
		// printf("Starting perf output, value size=%d, max entries = %d\n",
		//        map_info.value_size, map_info.max_entries);
		auto entry_size = sizeof(ringbuf_header) +
				  map_info.max_entries * (sizeof(uint64_t) +
							  map_info.value_size);
		auto header =
			(ringbuf_header *)(uintptr_t)(getGlobalThreadId() *
							      entry_size +
						      (char *)map_info
							      .extra_buffer);
		// printf("header->head=%lu, header->tail=%lu\n", header->head,
		//        header->tail);
		// NOTE: Avoid atomics on mapped host memory (sm_52 devices fault),
		// rely on per-thread buffers + dirty flag for synchronization.
		const uint64_t head_snapshot = header->head;
		const uint64_t tail_snapshot = header->tail;
		if (tail_snapshot - head_snapshot == map_info.max_entries) {
			// Buffer is full
			// printf("Buffer is full\n");
			return 2;
		}
		header->dirty = 1;
		__threadfence_system();
		auto real_tail = tail_snapshot % map_info.max_entries;
		// printf("real tail=%lu\n", real_tail);
		auto buffer =
			((char *)header) + sizeof(ringbuf_header) +
			real_tail * (sizeof(uint64_t) + map_info.value_size);
		// printf("before wrtting size to %lx, of %lu\n",
		//        (uintptr_t)buffer, data_size);
		*(uint64_t *)(uintptr_t)buffer = data_size;
		// printf("before copying..\n");
		simple_memcpy(buffer + sizeof(uint64_t),
			      (void *)(uintptr_t)data, data_size);
		// printf("data copied\n");
		__threadfence_system();
		header->tail = tail_snapshot + 1;
		__threadfence_system();
		header->dirty = 0;
		// printf("Generated %d bytes of data\n", (int)data_size);
		return 0;

	} else {
		printf("Calling bpf_perf_event_output on unsupported map!");
		return 1;
	}
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
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0506(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	asm("membar.sys;                      \n\t");
	return 0;
}
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0507(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	asm("exit;                      \n\t");
	return 0;
}
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0508(uint64_t x, uint64_t y, uint64_t z, uint64_t, uint64_t)
{
	// get grid dim
	*(uint64_t *)(uintptr_t)x = gridDim.x;
	*(uint64_t *)(uintptr_t)y = gridDim.y;
	*(uint64_t *)(uintptr_t)z = gridDim.z;

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
