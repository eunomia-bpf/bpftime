#include <__clang_cuda_builtin_vars.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
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
// Must match BPFTIME_GPU_HELPER_MAX_BUF in runtime/include/bpf_attach_ctx.hpp.
static constexpr int GPU_HELPER_MAX_BUF = 1 << 20;

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
		char key[GPU_HELPER_MAX_BUF];
	} map_lookup;
	struct {
		char key[GPU_HELPER_MAX_BUF];
		char value[GPU_HELPER_MAX_BUF];
		uint64_t flags;
	} map_update;
	struct {
		char key[GPU_HELPER_MAX_BUF];
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
static_assert(GPU_HELPER_MAX_BUF == 1 << 20,
	      "GPU helper host/device ABI requires a 1 MiB staging buffer");
static_assert(offsetof(CommSharedMem, req) == 24,
	      "unexpected GPU helper request offset");
static_assert(offsetof(CommSharedMem, resp) == 2097184,
	      "unexpected GPU helper response offset");
static_assert(sizeof(CommSharedMem) == 2097256,
	      "unexpected GPU helper shared-memory size");

const int BPF_MAP_TYPE_GPU_HASH_MAP = 1501; // non-per-thread, single-copy shared hashmap
// IPC-based GPU maps (for x86 with CUDA IPC support)
const int BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP = 1502;
const int BPF_MAP_TYPE_GPU_ARRAY_MAP = 1503; // non-per-thread, single-copy
					     // shared array
const int BPF_MAP_TYPE_GPU_RINGBUF_MAP = 1527;
const int BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP = 1504;

// HOST-based GPU maps (for Tegra/platforms without CUDA IPC)
const int BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP = 1512;
const int BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP = 1513;

struct MapBasicInfo {
	bool enabled;
	int key_size;
	int value_size;
	int max_entries;
	int map_type;
	void *extra_buffer;
	uint64_t max_thread_count;
	// Opt-in ring-output transport level for per-thread GPU rings.
	// 0 = legacy byte-wise copy over the legacy handshake, 1 = aligned
	// 8-byte payload copy over the legacy handshake, 2 = aligned copy
	// plus encoded tail publication in the dirty word, 3 = that
	// publication with the record words transposed so the host
	// reassembles each record into its bounded buffer. Reservation and
	// publication remain per event in every mode.
	uint64_t batch_output;
};
__device__ __forceinline__ uint64_t read_globaltimer()
{
	uint64_t timestamp;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
	return timestamp;
}

__constant__ uintptr_t constData;
__constant__ MapBasicInfo map_info[1024];
__device__ int __bpftime_comm_lock = 0;
extern "C" __device__ void spin_lock(volatile int *lock)
{
	while (atomicCAS((int *)lock, 0, 1) == 1) {
	}
	// printf("lock acquired by %d\n", threadIdx.x + blockIdx.x *
	// blockDim.x);
}

extern "C" __device__ void spin_unlock(int *lock)
{
	atomicExch(lock, 0);
	// printf("lock released by %d\n", threadIdx.x + blockIdx.x *
	// blockDim.x);
}

extern "C" __device__ inline void simple_memcpy(void *dst, void *src, int sz)
{
	for (int i = 0; i < sz; i++)
		((char *)dst)[i] = ((char *)src)[i];
}

// Copy aligned payloads in 8-byte words, retaining byte copies for
// unaligned addresses and trailing bytes. This is not cross-lane
// coalescing or a guarantee about hardware transaction counts.
extern "C" __device__ __forceinline__ void
coalesced_memcpy(void *dst, void *src, uint64_t sz)
{
	if (((uintptr_t)dst & 7u) == 0 && ((uintptr_t)src & 7u) == 0) {
		auto *d = (uint64_t *)dst;
		const auto *s = (const uint64_t *)src;
		uint64_t i = 0;
		for (; i + 8 <= sz; i += 8)
			d[i >> 3] = s[i >> 3];
		const char *cs = (const char *)src + i;
		char *cd = (char *)dst + i;
		for (; i < sz; i++)
			*cd++ = *cs++;
		return;
	}
	simple_memcpy(dst, src, (int)sz);
}

__device__ __noinline__ HelperCallResponse complete_helper_call_locked(
	CommSharedMem *g_data, long map_id, int req_id)
{
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
		     : "r"(val), "l"(&g_data->flag1), "l"(&g_data->flag2)
		     : "memory");

	return g_data->resp;
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
			spin_lock(&__bpftime_comm_lock);
			my_resp = complete_helper_call_locked(g_data, map_id,
							     req_id);
			spin_unlock(&__bpftime_comm_lock);
		}

		__syncwarp(active_mask);
	}

	return my_resp;
}

extern "C" __device__ HelperCallResponse make_map_helper_call(
	long map_id, int req_id, const void *key, int key_size,
	const void *value, int value_size, uint64_t flags)
{
	CommSharedMem *g_data = (CommSharedMem *)constData;
	int lane_id = threadIdx.x & 31;
	HelperCallResponse my_resp = {};

	if (key_size < 0 || key_size > GPU_HELPER_MAX_BUF ||
	    value_size < 0 || value_size > GPU_HELPER_MAX_BUF ||
	    (req_id != (int)HelperOperation::MAP_LOOKUP &&
	     req_id != (int)HelperOperation::MAP_UPDATE &&
	     req_id != (int)HelperOperation::MAP_DELETE))
		return my_resp;

	for (int active_lane = 0; active_lane < 32; active_lane++) {
		unsigned int active_mask = __activemask();
		bool lane_is_active = (active_mask >> active_lane) & 1;

		if (lane_is_active && lane_id == active_lane) {
			spin_lock(&__bpftime_comm_lock);
			if (req_id == (int)HelperOperation::MAP_LOOKUP) {
				simple_memcpy(g_data->req.map_lookup.key,
					      (void *)key, key_size);
			} else if (req_id == (int)HelperOperation::MAP_UPDATE) {
				simple_memcpy(g_data->req.map_update.key,
					      (void *)key, key_size);
				simple_memcpy(g_data->req.map_update.value,
					      (void *)value, value_size);
				g_data->req.map_update.flags = flags;
			} else if (req_id == (int)HelperOperation::MAP_DELETE) {
				simple_memcpy(g_data->req.map_delete.key,
					      (void *)key, key_size);
			}
			my_resp = complete_helper_call_locked(g_data, map_id,
							     req_id);
			spin_unlock(&__bpftime_comm_lock);
		}

		__syncwarp(active_mask);
	}

	return my_resp;
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
__device__ void *array_map_offset(uint64_t idx, const MapBasicInfo &info,
				  uint64_t map)
{
	if (info.max_thread_count <= getGlobalThreadId()) {
		printf("WARNING: getGlobalThreadId(%lu) exceeds max_thread_count (%lu) of map %lu, please set BPFTIME_MAP_GPU_THREAD_COUNT at syscall-server side\n",
		       getGlobalThreadId(), info.max_thread_count, map);
	}
	return (void *)((uintptr_t)info.extra_buffer +
			idx * info.max_thread_count * info.value_size +
			getGlobalThreadId() * info.value_size);
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0001(
	uint64_t map, uint64_t key, uint64_t a, uint64_t b, uint64_t c)
{
	const auto &map_info = ::map_info[map];
	// IPC-based per-thread array map
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info, map);
		return (uint64_t)offset;
	}
	// Fast-path for non-per-thread GPU array map: single shared copy on
	// device-visible UVA
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_MAP ||
	    map_info.map_type == BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		// printf("real_key=%u, base=%lx, value_size=%lu, mapfd=%d\n",real_key,(uintptr_t)base,(unsigned long)map_info.value_size,(int)map);
		return (uint64_t)(uintptr_t)(base +
					     (uint64_t)real_key *
						     map_info.value_size);
	}
	// HOST-based per-thread array map (for Tegra)
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info, map);
		asm("membar.sys;"); // Ensure CPU writes are visible to GPU
		return (uint64_t)offset;
	}
	// HOST-based non-per-thread GPU array map (for Tegra)
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		asm("membar.sys;"); // Ensure CPU writes are visible to GPU
		return (uint64_t)(uintptr_t)(base + (uint64_t)real_key * map_info.value_size);
	}
	HelperCallResponse resp =
		make_map_helper_call((long)map,
				     (int)HelperOperation::MAP_LOOKUP,
				     (void *)(uintptr_t)key, map_info.key_size,
				     nullptr, 0, 0);

	return (uintptr_t)resp.map_lookup.value;
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0002(
	uint64_t map, uint64_t key, uint64_t value, uint64_t flags, uint64_t a)
{
	const auto &map_info = ::map_info[map];
	// IPC-based per-thread array map
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info, map);
		simple_memcpy(offset, (void *)(uintptr_t)value,
			      map_info.value_size);
		return 0;
	}
	// Fast-path for non-per-thread GPU array map: memcpy overwrite, system
	// fence for visibility
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_MAP ||
	    map_info.map_type == BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		auto dst =
			(void *)(uintptr_t)(base + (uint64_t)real_key *
							   map_info.value_size);
		simple_memcpy(dst, (void *)(uintptr_t)value,
			      map_info.value_size);
		asm("membar.sys;                      \n\t");
		return 0;
	}
	// HOST-based per-thread array map (for Tegra)
	if (map_info.map_type == BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto offset = array_map_offset(real_key, map_info, map);
		simple_memcpy(offset, (void *)(uintptr_t)value,
			      map_info.value_size);
		asm("membar.sys;"); // Ensure GPU writes are visible to CPU
		return 0;
	}
	// HOST-based non-per-thread GPU array map (for Tegra)
	if (map_info.map_type == BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP) {
		auto real_key = *(uint32_t *)(uintptr_t)key;
		auto base = (char *)map_info.extra_buffer;
		auto dst = (void *)(uintptr_t)(base + (uint64_t)real_key * map_info.value_size);
		simple_memcpy(dst, (void *)(uintptr_t)value, map_info.value_size);
		asm("membar.sys;"); // Ensure GPU writes are visible to CPU
		return 0;
	}
	HelperCallResponse resp =
		make_map_helper_call((long)map,
				     (int)HelperOperation::MAP_UPDATE,
				     (void *)(uintptr_t)key, map_info.key_size,
				     (void *)(uintptr_t)value,
				     map_info.value_size, flags);
	return resp.map_update.result;
}

extern "C" __noinline__ __device__ uint64_t _bpf_helper_ext_0003(
	uint64_t map, uint64_t key, uint64_t a, uint64_t b, uint64_t c)
{
	const auto &map_info = ::map_info[map];
	HelperCallResponse resp =
		make_map_helper_call((long)map,
				     (int)HelperOperation::MAP_DELETE,
				     (void *)(uintptr_t)key, map_info.key_size,
				     nullptr, 0, 0);
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
	uint64_t dirty;
};
static_assert(alignof(ringbuf_header) == alignof(uint64_t),
	      "GPU ring-buffer header must retain 64-bit alignment");
static_assert(offsetof(ringbuf_header, head) == 0 &&
		      offsetof(ringbuf_header, tail) == 8 &&
		      offsetof(ringbuf_header, dirty) == 16 &&
		      sizeof(ringbuf_header) == 24,
	      "unexpected GPU ring-buffer header layout");

struct ringbuf_error_counters {
	uint64_t oob_drops;
	uint64_t full_drops;
	uint64_t bad_size_drops;
	uint64_t other_drops;
};

// Bit 63 of bpf_perf_event_output flags: opt-in GPU-only per-active-warp
// output. When set, only the first active lane of the calling warp runs the
// ring-buffer reserve/copy path; all other active lanes return 0.
static constexpr uint64_t BPF_PERF_EVENT_GPU_WARP_ONLY_OUTPUT = 1ull << 63;

// perf event output
extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0025(uint64_t ctx, uint64_t map, uint64_t flags, uint64_t data,
		     uint64_t data_size)
{
	const auto &map_info = ::map_info[map];
	if (map_info.map_type == BPF_MAP_TYPE_GPU_RINGBUF_MAP) {
		if (flags & BPF_PERF_EVENT_GPU_WARP_ONLY_OUTPUT) {
			const unsigned int active_mask = __activemask();
			const int first_active_lane = __ffs(active_mask) - 1;
			uint32_t lane_id;
			asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
			if ((int)lane_id != first_active_lane)
				return 0;
		}
		const auto record_stride =
			(sizeof(uint64_t) + map_info.value_size +
			 sizeof(uint64_t) - 1) &
			~(sizeof(uint64_t) - 1);
		const auto entry_size = sizeof(ringbuf_header) +
			map_info.max_entries * record_stride;
		auto *errors = (ringbuf_error_counters *)(uintptr_t)(
			(char *)map_info.extra_buffer +
			map_info.max_thread_count * entry_size);
		const auto tid = getGlobalThreadId();
		if (tid >= map_info.max_thread_count) {
			atomicAdd_system(
				(unsigned long long *)&errors->oob_drops, 1);
			return 1;
		}
		if (data_size > map_info.value_size) {
			atomicAdd_system(
				(unsigned long long *)&errors->bad_size_drops, 1);
			return 3;
		}
		if (data_size != 0 && data == 0) {
			atomicAdd_system(
				(unsigned long long *)&errors->other_drops, 1);
			return 4;
		}
		if (map_info.batch_output == 3) {
			// Transposed record transport. The 24-byte headers
			// are packed together and the record words are
			// transposed so that, for a fixed slot and word, the
			// thread-major stride is one 8-byte word: adjacent
			// thread ids write adjacent 8-byte locations. This is
			// a record-preserving layout change only, not a change
			// to the per-event publication protocol: encoded dirty
			// publication, head read, capacity check, and the
			// return/drop colors are unchanged from the
			// encoded-release mode. Word 0 carries the payload
			// size; the host reassembles each record into its
			// bounded buffer before delivery.
			const uint64_t n = map_info.max_thread_count;
			const uint64_t record_words =
				(uint64_t)(record_stride / sizeof(uint64_t));
			auto *header =
				(ringbuf_header *)(uintptr_t)(tid *
							      sizeof(ringbuf_header) +
						      (char *)map_info
							      .extra_buffer);
			const uint64_t pre =
				*(volatile uint64_t *)(uintptr_t)&header->dirty;
			if ((pre & 1) != 0) {
				atomicAdd_system(
					(unsigned long long *)&errors->other_drops, 1);
				return 4;
			}
			if (atomicCAS_system((unsigned long long *)&header->dirty,
					     pre, pre | 1) != pre) {
				atomicAdd_system(
					(unsigned long long *)&errors->other_drops, 1);
				return 4;
			}
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t head =
				*(volatile uint64_t *)(uintptr_t)&header->head;
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t tail = pre >> 1;
			if (tail - head >= (uint64_t)map_info.max_entries) {
				atomicAdd_system((unsigned long long *)&errors
						 ->full_drops,
						1);
				asm volatile("membar.sys;" ::: "memory");
				atomicExch_system((unsigned long long *)&header->dirty,
						  pre);
				return 2;
			}
			const uint64_t real_tail =
				tail % (uint64_t)map_info.max_entries;
			// Packed headers end at n * 24; the transposed word
			// for slot s, word w, thread tid sits at
			// payload_base + ((s * record_words + w) * n + tid) * 8.
			char *payload_base =
				(char *)map_info.extra_buffer +
				n * sizeof(ringbuf_header);
			const uint64_t word_stride = n * sizeof(uint64_t);
			const uint64_t slot_base =
				(real_tail * record_words) * word_stride +
				tid * sizeof(uint64_t);
			// word 0 = payload size
			*(uint64_t *)(uintptr_t)(payload_base + slot_base) =
				data_size;
			// words 1..record_words-1 = payload, zero-padded
			// tail; the source is read byte-wise so an
			// unaligned payload and a trailing partial word are
			// copied exactly.
			const unsigned char *src =
				(const unsigned char *)(uintptr_t)data;
			for (uint64_t w = 1; w < record_words; w++) {
				const uint64_t src_off = (w - 1) * 8;
				uint64_t v = 0;
				if (src_off < data_size) {
					uint64_t valid = data_size - src_off;
					if (valid > 8)
						valid = 8;
					for (uint64_t b = 0; b < valid; b++)
						v |= (uint64_t)src[src_off + b] <<
						       (8 * b);
				}
				*(uint64_t *)(uintptr_t)(payload_base + slot_base +
							   w * word_stride) =
					v;
			}
			asm volatile("membar.sys;" ::: "memory");
			atomicExch_system((unsigned long long *)&header->dirty,
					  (tail + 1) << 1);
			return 0;
		}
		if (map_info.batch_output == 2) {
		// Encoded-release protocol for this slot: a per-event
		// publication optimization, not warp-level batching of
		// multiple records and not a measured hardware-traffic
		// win. The unlocked dirty word carries the published tail
		// shifted left by one (odd = locked); the
		// protocol-level intent is one fewer system-scope RMW and
		// one fewer fence per event than the legacy two-exchange
		// publication, with the same publication strength, drop
		// colors, return codes, record layout, and per-thread slot
		// semantics. A crash while locked leaves the word odd, the
		// slot skipped forever by the drain and later calls of the
		// same thread dropped with the collision color, identical
		// to a leftover legacy lock.
			auto *header =
				(ringbuf_header *)(uintptr_t)(tid * entry_size +
							      (char *)map_info
								      .extra_buffer);
			const uint64_t pre =
				*(volatile uint64_t *)(uintptr_t)&header->dirty;
			if ((pre & 1) != 0) {
				atomicAdd_system(
					(unsigned long long *)&errors->other_drops, 1);
				return 4;
			}
			if (atomicCAS_system((unsigned long long *)&header->dirty,
					     pre, pre | 1) != pre) {
				atomicAdd_system(
					(unsigned long long *)&errors->other_drops, 1);
				return 4;
			}
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t head =
				*(volatile uint64_t *)(uintptr_t)&header->head;
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t tail = pre >> 1;
			if (tail - head >= (uint64_t)map_info.max_entries) {
				atomicAdd_system((unsigned long long *)&errors
							 ->full_drops,
						 1);
				asm volatile("membar.sys;" ::: "memory");
				atomicExch_system((unsigned long long *)&header->dirty,
						  pre);
				return 2;
			}
			const auto real_tail = tail % (uint64_t)map_info.max_entries;
			auto buffer = ((char *)header) + sizeof(ringbuf_header) +
				      real_tail * record_stride;
			*(uint64_t *)(uintptr_t)buffer = data_size;
			coalesced_memcpy(buffer + sizeof(uint64_t),
					 (void *)(uintptr_t)data, data_size);
			asm volatile("membar.sys;" ::: "memory");
			atomicExch_system((unsigned long long *)&header->dirty,
					  (tail + 1) << 1);
			return 0;
		}
		if (map_info.batch_output == 1) {
			// Opt-in transport fast path. The per-thread
			// CAS/publication/drop semantics are identical to the
			// legacy path below: same slot, same dirty lock with
			// the same collision drop, same full-slot drop rule,
			// same head/tail protocol, same return codes and
			// record layout. Only the payload copy uses aligned
			// 8-byte words where possible; performance and hardware
			// transaction counts require measurement.
			auto *header =
				(ringbuf_header *)(uintptr_t)(tid * entry_size +
							      (char *)map_info
								      .extra_buffer);
			if (atomicCAS_system(
				    (unsigned long long *)&header->dirty, 0,
				    1) != 0) {
				atomicAdd_system(
					(unsigned long long *)&errors->other_drops,
					1);
				return 4;
			}
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t head =
				*(volatile uint64_t *)(uintptr_t)&header->head;
			asm volatile("membar.sys;" ::: "memory");
			const uint64_t tail = header->tail;
			if (tail - head >= (uint64_t)map_info.max_entries) {
				atomicAdd_system(
					(unsigned long long *)&errors->full_drops,
					1);
				asm volatile("membar.sys;" ::: "memory");
				atomicExch_system(
					(unsigned long long *)&header->dirty,
					0);
				return 2;
			}
			const auto real_tail =
				tail % (uint64_t)map_info.max_entries;
			auto buffer = ((char *)header) + sizeof(ringbuf_header) +
				      real_tail * record_stride;
			*(uint64_t *)(uintptr_t)buffer = data_size;
			coalesced_memcpy(buffer + sizeof(uint64_t),
					 (void *)(uintptr_t)data, data_size);
			asm volatile("membar.sys;" ::: "memory");
			atomicExch_system((unsigned long long *)&header->tail,
					  tail + 1);
			asm volatile("membar.sys;" ::: "memory");
			atomicExch_system((unsigned long long *)&header->dirty,
					  0);
			return 0;
		}
		// printf("Starting perf output, value size=%d, max entries =
		// %d\n",
		//        map_info.value_size, map_info.max_entries);
		auto header =
			(ringbuf_header *)(uintptr_t)(tid * entry_size +
						      (char *)map_info
							      .extra_buffer);
		if (atomicCAS_system((unsigned long long *)&header->dirty, 0, 1) !=
		    0) {
			atomicAdd_system(
				(unsigned long long *)&errors->other_drops, 1);
			return 4;
		}
		asm volatile("membar.sys;" ::: "memory");
		// printf("header->head=%lu, header->tail=%lu\n", header->head,
		//        header->tail);
		const uint64_t head =
			*(volatile uint64_t *)(uintptr_t)&header->head;
		asm volatile("membar.sys;" ::: "memory");
		const uint64_t tail = header->tail;
		if (tail - head >= map_info.max_entries) {
			// Buffer is full
			// printf("Buffer is full\n");
			atomicAdd_system(
				(unsigned long long *)&errors->full_drops, 1);
			asm volatile("membar.sys;" ::: "memory");
			atomicExch_system(
				(unsigned long long *)&header->dirty, 0);
			return 2;
		}
		const auto real_tail = tail % map_info.max_entries;
		// printf("real tail=%lu\n", real_tail);
		auto buffer =
			((char *)header) + sizeof(ringbuf_header) +
			real_tail * record_stride;
		// printf("before wrtting size to %lx, of %lu\n",
		//        (uintptr_t)buffer, data_size);
		*(uint64_t *)(uintptr_t)buffer = data_size;
		// printf("before copying..\n");
		simple_memcpy(buffer + sizeof(uint64_t),
			      (void *)(uintptr_t)data, data_size);
		// printf("data copied\n");
		asm volatile("membar.sys;" ::: "memory");
		atomicExch_system((unsigned long long *)&header->tail, tail + 1);
		asm volatile("membar.sys;" ::: "memory");
		atomicExch_system((unsigned long long *)&header->dirty, 0);
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

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0509(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
    // get sm id
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return (uint64_t)sm_id;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0510(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
    // get warp id
    uint32_t warp_id;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    return (uint64_t)warp_id;
}

extern "C" __noinline__ __device__ uint64_t
_bpf_helper_ext_0511(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
    // get lane id
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return (uint64_t)lane_id;
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
