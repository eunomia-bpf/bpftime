#include "bpf/bpf.h"
#include "bpf/libbpf_common.h"
#include "bpftool/libbpf/src/libbpf.h"
#include "linux/bpf_common.h"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <bpf_map/shared/perf_event_array_kernel_user.hpp>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <linux/bpf.h>
#include <map>
#include <memory>
#include <pthread.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <linux/filter.h>
#include <sys/ioctl.h>
#include <bpf/libbpf.h>

// kernel perf event array map id -> user_ring_buffer* for the current process
using user_ringbuf_map =
	std::map<int, std::unique_ptr<bpftime::user_ringbuffer_wrapper> >;
user_ringbuf_map *user_rb_map = nullptr;
static inline void ensure_user_rb_map_initialized()
{
	if (!user_rb_map)
		user_rb_map = new user_ringbuf_map;
}
namespace bpftime
{
// Wraps a user_ringbuffer and it's spinlock that locks the reservation
struct user_ringbuffer_wrapper {
	pthread_spinlock_t reserve_lock;
	user_ring_buffer *rb;
	int user_rb_id;
	int user_rb_fd;
	user_ringbuffer_wrapper(int user_rb_id);
	~user_ringbuffer_wrapper();
	void *reserve(uint32_t size);
	void submit(void *mem);
};

int perf_event_array_kernel_user_impl::get_user_ringbuf_fd()
{
	return ensure_current_map_user_ringbuf()->user_rb_fd;
}
int perf_event_array_kernel_user_impl::output_data_into_kernel(const void *buf,
							       size_t size)
{
	spdlog::debug("Received data output for kernel perf event array {}",
		      kernel_perf_id);
	auto user_rb = ensure_current_map_user_ringbuf();
	spdlog::debug("User ringbuf ensured: {:x}", (uintptr_t)user_rb);
	void *mem = user_rb->reserve(size + 8);
	if (!mem) {
		spdlog::error("Failed to reserve for user ringbuf: {}", errno);
		return errno;
	}
	*(uint64_t *)(mem) = (uint64_t)size;
	memcpy((char *)mem + 8, buf, size);
	user_rb->submit(mem);
	spdlog::trace("Commited {} bytes of data into kernel", size);
	return 0;
}
// Put the creation of user ringbuffer & transporter ebpf program in the
// constructor of perf_event_array_kernel_user_impl Only one instance of map and
// ebpf program is required, so just put them in the daemon
perf_event_array_kernel_user_impl::perf_event_array_kernel_user_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t key_size,
	uint32_t value_size, uint32_t max_entries, int kernel_perf_id)
	: max_ent(max_entries), kernel_perf_id(kernel_perf_id)
{
	if (key_size != 4 || value_size != 4) {
		spdlog::error(
			"Key size and value size of perf_event_array must be 4");
		assert(false);
	}
	// Create corresponding user ringbuffer
	LIBBPF_OPTS(bpf_map_create_opts, user_rb_opts);
	std::string name = "ku_perf_id_" + std::to_string(kernel_perf_id);
	int user_rb_fd = bpf_map_create(BPF_MAP_TYPE_USER_RINGBUF, name.c_str(),
					0, 0, 1024 * 1024, &user_rb_opts);
	if (user_rb_fd < 0) {
		spdlog::error(
			"Failed to create user ringbuffer for shared perf event array id {}, err={}",
			kernel_perf_id, errno);
		return;
	}
	bpf_map_info map_info;
	uint32_t map_info_size = sizeof(map_info);
	if (int err = bpf_obj_get_info_by_fd(user_rb_fd, &map_info,
					     &map_info_size);
	    err < 0) {
		spdlog::error("Failed to get map info for user rb fd {}",
			      user_rb_fd);
		return;
	}
	user_rb_id = map_info.id;
	spdlog::debug(
		"Initialized perf_event_array_kernel_user_impl, kernel perf id {}, user ringbuffer id {}, user ringbuffer map type {}",
		kernel_perf_id, user_rb_id, (int)map_info.type);
	int pfd = create_intervally_triggered_perf_event(10);
    int kernel_perf_fd = bpf_map_get_fd_by_id(kernel_perf_id);
	auto prog = create_transporting_kernel_ebpf_program(user_rb_fd,
							    kernel_perf_fd);
	std::vector<int> fds;
	fds.push_back(user_rb_fd);
	fds.push_back(kernel_perf_fd);

	LIBBPF_OPTS(bpf_prog_load_opts, opts);
	char log_buffer[2048];
	opts.log_buf = log_buffer;
	opts.log_size = sizeof(log_buffer);
	opts.log_level = 5;
	opts.fd_array = fds.data();

	spdlog::debug("Loading transporter program with {} insns", prog.size());
	int bpf_fd =
		bpf_prog_load(BPF_PROG_TYPE_PERF_EVENT, "transporter", "GPL",
			      (bpf_insn *)prog.data(), prog.size(), &opts);
	if (bpf_fd < 0) {
		spdlog::error("Failed to load bpf prog: err={}, message={}",
			      errno, log_buffer);
	}
	assert(bpf_fd >= 0);
	int err;
	err = ioctl(pfd, PERF_EVENT_IOC_SET_BPF, bpf_fd);
	if (err < 0) {
		spdlog::error("Failed to run PERF_EVENT_IOC_SET_BPF: {}", err);
		assert(false);
	}
	err = ioctl(pfd, PERF_EVENT_IOC_ENABLE, 0);
	if (err < 0) {
		spdlog::error("Failed to run PERF_EVENT_IOC_ENABLE: {}", err);
		assert(false);
	}
	spdlog::debug("Attached transporter ebpf program");
}
perf_event_array_kernel_user_impl::~perf_event_array_kernel_user_impl()
{
}

void *perf_event_array_kernel_user_impl::elem_lookup(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return nullptr;
	}
	spdlog::info(
		"Looking up key {} from perf event array kernel user, which is useless",
		k);
	return &dummy;
}

long perf_event_array_kernel_user_impl::elem_update(const void *key,
						    const void *value,
						    uint64_t flags)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	int32_t v = *(int32_t *)value;
	spdlog::info(
		"Updating key {}, value {} from perf event array kernel user, which is useless",
		k, v);
	return 0;
}

long perf_event_array_kernel_user_impl::elem_delete(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	spdlog::info(
		"Deleting key {} from perf event array kernel user, which is useless",
		k);
	return 0;
}

int perf_event_array_kernel_user_impl::map_get_next_key(const void *key,
							void *next_key)
{
	int32_t *out = (int32_t *)next_key;
	if (key == nullptr) {
		*out = 0;
		return 0;
	}
	int32_t k = *(int32_t *)key;
	// The last key
	if ((size_t)(k + 1) == max_ent) {
		errno = ENOENT;
		return -1;
	}
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return -1;
	}
	*out = k + 1;
	return 0;
}
user_ringbuffer_wrapper *
perf_event_array_kernel_user_impl::ensure_current_map_user_ringbuf()
{
	ensure_user_rb_map_initialized();
	user_ringbuffer_wrapper *result;
	if (auto itr = user_rb_map->find(kernel_perf_id);
	    itr != user_rb_map->end()) {
		result = itr->second.get();
	} else {
		auto ptr = std::make_unique<user_ringbuffer_wrapper>(
			kernel_perf_id);
		auto raw_ptr = ptr.get();
		user_rb_map->emplace(kernel_perf_id, std::move(ptr));
		result = raw_ptr;
	}
	return result;
}
user_ringbuffer_wrapper::user_ringbuffer_wrapper(int user_rb_id)
	: user_rb_id(user_rb_id)
{
	pthread_spin_init(&reserve_lock, PTHREAD_PROCESS_PRIVATE);

	LIBBPF_OPTS(user_ring_buffer_opts, opts);
	user_rb_fd = bpf_map_get_fd_by_id(user_rb_id);
	spdlog::debug("map id {} -> fd {}, user ring buffer", user_rb_id,
		      user_rb_fd);
	if (user_rb_fd < 0) {
		spdlog::error(
			"Failed to get user_rb_fd from user_rb_id {}, err={}",
			user_rb_id, errno);
		throw std::runtime_error(
			"Failed to get user_rb_fd from user_rb_id");
	}
	rb = user_ring_buffer__new(user_rb_fd, &opts);
	assert(rb &&
	       "Failed to initialize user ringbuffer! This SHOULD NOT Happen.");
	spdlog::debug("User ringbuffer wrapper created, fd={}, id={}",
		      user_rb_fd, user_rb_id);
}

user_ringbuffer_wrapper::~user_ringbuffer_wrapper()
{
	pthread_spin_destroy(&reserve_lock);
	user_ring_buffer__free(rb);
}
void *user_ringbuffer_wrapper::reserve(uint32_t size)
{
	return user_ring_buffer__reserve(rb, (uint32_t)size);
}
void user_ringbuffer_wrapper::submit(void *mem)
{
	user_ring_buffer__submit(rb, mem);
}

// The original program was like:
/*
#define NULL 0
static long (*bpf_dynptr_read)(void *dst, int len, const struct bpf_dynptr *src,
int offset, long flags) = (void *) 201; static long
(*bpf_perf_event_output)(void *ctx, void *map, long flags, void *data, long
size) = (void *) 25; static long (*bpf_user_ringbuf_drain)(void *map, void
*callback_fn, void *ctx, long flags) = (void *) 209;

static int cb(void *dynptr, void *ctx);

int func(void* ctx)
{
	bpf_user_ringbuf_drain((void*)0x234, &cb, ctx, 0);
    return 0;
}
static int cb(void *dynptr, void *ctx)
{
	long val;
    char buf[496];
	bpf_dynptr_read(&val, 8, dynptr, 0, 0);
	if(val>400) return 1;
	bpf_dynptr_read(&buf, val, dynptr, 8, 0);
	bpf_perf_event_output(ctx, (void*)0x123, 0, buf, val);
	return 1;
}
*/

// Compiled using gcc.godbolt.org

std::vector<uint64_t>
create_transporting_kernel_ebpf_program(int user_ringbuf_fd,
					int perf_event_array_fd)
{
	static_assert(
		sizeof(bpf_insn) == sizeof(uint64_t),
		"bpf_insn is expected to be in the same size of uint64_t");
	bpf_insn insns[] = {
		// r3 = r1
		BPF_MOV64_REG(3, 1),
		// r1 = map_by_fd(user_ringbuf_fd)
		BPF_LD_IMM64_RAW_FULL(1, 1, 0, 0, user_ringbuf_fd, 0),
		// r2 = callback fn
		BPF_MOV64_IMM(2, 64),
		// r4 = 0
		BPF_MOV64_IMM(4, 0),
		// call bpf_user_ringbuf_drain(void *map, void *callback_fn,
		// void *ctx, long flags) = 0x209
		BPF_EMIT_CALL(0x209),
		// r0 = 0
		BPF_MOV64_IMM(0, 0), BPF_EXIT_INSN(),
		// static int cb(void *dynptr, void *ctx)
		// r6 = r2
		BPF_MOV64_REG(6, 2),
		// r7 = r1
		BPF_MOV64_REG(7, 1),
		// r1 = r10
		BPF_MOV64_REG(1, 10),
		// r1 += -0x8
		BPF_ALU64_IMM(BPF_ADD, 1, -8),
		// r2=8,
		BPF_MOV64_IMM(2, 8),
		// r3 = r7
		BPF_MOV64_IMM(3, 7),
		// r4=0
		BPF_MOV64_IMM(4, 0),
		// r5 = 0
		BPF_MOV64_IMM(5, 0),
		// call 0xc9 bpf_dynptr_read(void *dst, int len, const struct
		// bpf_dynptr *src, int offset, long flags) = (void *) 201
		BPF_EMIT_CALL(0xc9),
		// r2 = *(u64 *)(r10 - 0x8)
		BPF_LDX_MEM(BPF_DW, 2, 10, -8),
		// if r2 s> 0x1f0 goto +0xd <LBB1_2>
		BPF_RAW_INSN(0x65, 0x2, 0x0, 0x0d, 0x1f0),
		// r8 = r10
		BPF_MOV64_REG(8, 10),
		// r8 += -0x1f8
		BPF_ALU64_IMM(BPF_ADD, 8, -0x1f8),
		// r1 = r8
		BPF_MOV64_REG(1, 8),
		// r3 = r7
		BPF_MOV64_REG(3, 7),
		// r4 = 8
		BPF_MOV64_IMM(4, 8),
		// r5 = 0
		BPF_MOV64_IMM(5, 0),
		// call 0xc9 bpf_dynptr_read(void *dst, int len, const struct
		// bpf_dynptr *src, int offset, long flags) = (void *) 201
		BPF_EMIT_CALL(0xc9),
		// r5 = *(u64 *)(r10 - 0x8)
		BPF_LDX_MEM(BPF_DW, 5, 10, -8),
		// r1 = r6
		BPF_MOV64_REG(1, 6),
		// r2 = map_by_fd(perf_event_array_fd)
		BPF_LD_IMM64_RAW_FULL(2, 1, 0, 0, perf_event_array_fd, 0),
		// r3 = 0
		BPF_MOV64_IMM(3, 0),
		// r4 = r8
		BPF_MOV64_REG(4, 8),
		// call 0x19 bpf_perf_event_output(void *ctx, void *map, long
		// flags, void *data, long size) = 25
		BPF_EMIT_CALL(25),
		// r0 = 1
		BPF_MOV64_IMM(0, 1), BPF_EXIT_INSN()
	};
	return std::vector<uint64_t>((uint64_t *)&insns,
				     (uint64_t *)&insns + std::size(insns));
}

int create_intervally_triggered_perf_event(int freq)
{
	perf_event_attr pe_attr = {
		.type = PERF_TYPE_SOFTWARE,
		.size = sizeof(struct perf_event_attr),
		.config = PERF_COUNT_SW_CPU_CLOCK,
		.sample_freq = (__u64)freq,
		.freq = 1,
	};
	return syscall(__NR_perf_event_open, &pe_attr, 0, -1, -1, 0);
}
} // namespace bpftime
