/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <linux/bpf.h>
#include <linux/perf_event.h>
#include <linux/btf.h>
#include "libbpf/include/linux/filter.h"
#include <linux/bpf_common.h>
#include <linux/perf_event.h>
#include <asm/unistd.h> 
#include <sys/ioctl.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <bpf/btf.h>
#include <bpf/libbpf_common.h>
#include <map>
#include <unistd.h>
#include <memory>
#include <pthread.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include "spdlog/fmt/bin_to_hex.h"
#include <bpf_map/shared/perf_event_array_kernel_user.hpp>
#include <cerrno>
#include <cstring>
#include <iterator>

static int create_transporter_prog(int user_ringbuf_fd, int kernel_perf_fd);
// Ubuntu 22.04's linux/bpf.h can lag behind libbpf and miss the
// BPF_MAP_TYPE_USER_RINGBUF enumerator even though libbpf supports it.
#ifndef BPF_MAP_TYPE_USER_RINGBUF
#define BPF_MAP_TYPE_USER_RINGBUF 31
#endif

namespace
{

struct perf_daemon_local_entry {
	int pfd;
	int transporter_prog_fd;
};

struct perf_user_rb_local_entry {
	user_ring_buffer *rb;
	int user_rb_fd;
};

bpftime::shm_local_table<perf_daemon_local_entry> g_perf_daemon_local_table{};
bpftime::shm_local_table<perf_user_rb_local_entry> g_perf_user_rb_local_table{};

static_assert(
	std::is_trivially_destructible_v<decltype(g_perf_daemon_local_table)>);
static_assert(
	std::is_trivially_destructible_v<decltype(g_perf_user_rb_local_table)>);

void cleanup_perf_daemon_local(perf_daemon_local_entry &entry) noexcept
{
	if (entry.pfd >= 0) {
		ioctl(entry.pfd, PERF_EVENT_IOC_DISABLE, 0);
		close(entry.pfd);
		entry.pfd = -1;
	}
	if (entry.transporter_prog_fd >= 0) {
		close(entry.transporter_prog_fd);
		entry.transporter_prog_fd = -1;
	}
}

void cleanup_perf_user_rb_local(perf_user_rb_local_entry &entry) noexcept
{
	if (entry.rb != nullptr) {
		user_ring_buffer__free(entry.rb);
		entry.rb = nullptr;
	}
	if (entry.user_rb_fd >= 0) {
		close(entry.user_rb_fd);
		entry.user_rb_fd = -1;
	}
}

perf_user_rb_local_entry *resolve_perf_user_rb_local(bpftime::shm_local_key key,
						     int user_rb_id)
{
	if (user_rb_id <= 0)
		return nullptr;
	return g_perf_user_rb_local_table.get_or_init(
		key,
		[&](perf_user_rb_local_entry &out) -> bool {
			LIBBPF_OPTS(user_ring_buffer_opts, opts);
			int fd = bpf_map_get_fd_by_id(user_rb_id);
			if (fd < 0) {
				const int saved_errno = errno;
				SPDLOG_ERROR(
					"Failed to get user_rb_fd from user_rb_id {}, err={}",
					user_rb_id, saved_errno);
				errno = saved_errno;
				return false;
			}
			user_ring_buffer *rb = user_ring_buffer__new(fd, &opts);
			if (rb == nullptr) {
				const int saved_errno = errno;
				close(fd);
				SPDLOG_ERROR(
					"Failed to initialize user ringbuffer for id {}",
					user_rb_id);
				errno = saved_errno;
				return false;
			}
			out.rb = rb;
			out.user_rb_fd = fd;
			return true;
		},
		cleanup_perf_user_rb_local);
}

} // namespace

namespace bpftime
{

int perf_event_array_kernel_user_impl::get_user_ringbuf_fd()
{
	auto *local = resolve_perf_user_rb_local(local_key_, user_rb_id);
	return local != nullptr ? local->user_rb_fd : -1;
}

int perf_event_array_kernel_user_impl::output_data_into_kernel(const void *buf,
							       size_t size)
{
	if (size > 480 - 32) {
		SPDLOG_ERROR(
			"Max data size for shared perf event array is {} bytes",
			480 - 32);
		errno = E2BIG;
		return -1;
	}
	auto *user_rb = resolve_perf_user_rb_local(local_key_, user_rb_id);
	if (user_rb == nullptr || user_rb->rb == nullptr) {
		errno = EINVAL;
		return -1;
	}
	void *mem = user_ring_buffer__reserve(user_rb->rb,
					      static_cast<uint32_t>(size + 8));
	if (!mem) {
		SPDLOG_ERROR("Failed to reserve for user ringbuf: {}", errno);
		return errno;
	}
	*(uint64_t *)(mem) = (uint64_t)size;
	memcpy((char *)mem + 8, buf, size);
	user_ring_buffer__submit(user_rb->rb, mem);
	return 0;
}

perf_event_array_kernel_user_impl::perf_event_array_kernel_user_impl(
	const shm_init_ctx &ctx, uint32_t key_size, uint32_t value_size,
	uint32_t max_entries, int kernel_perf_id)
	: max_ent(max_entries), kernel_perf_id(kernel_perf_id),
	  seg_alloc_(ctx.allocator<char>())
{
	if (key_size != 4 || value_size != 4) {
		SPDLOG_ERROR(
			"Key size and value size of perf_event_array must be 4");
		throw std::runtime_error(
			"Key size and value size of perf_event_array must be 4");
	}
	LIBBPF_OPTS(bpf_map_create_opts, user_rb_opts);
	std::string name = "ku_perf_id_" + std::to_string(kernel_perf_id);
	int user_rb_fd = bpf_map_create(
		static_cast<enum bpf_map_type>(BPF_MAP_TYPE_USER_RINGBUF),
		name.c_str(), 0, 0, 1024 * 1024, &user_rb_opts);
	if (user_rb_fd < 0) {
		SPDLOG_ERROR(
			"Failed to create user ringbuffer for shared perf event array id {}, err={}",
			kernel_perf_id, errno);
		return;
	}
	bpf_map_info map_info{};
	uint32_t map_info_size = sizeof(map_info);
	if (bpf_obj_get_info_by_fd(user_rb_fd, &map_info, &map_info_size) < 0) {
		SPDLOG_ERROR("Failed to get map info for user rb fd {}",
			     user_rb_fd);
		close(user_rb_fd);
		return;
	}
	user_rb_id = map_info.id;
	int pfd = create_intervally_triggered_perf_event(10);
	int kernel_perf_fd = bpf_map_get_fd_by_id(kernel_perf_id);
	int bpf_fd = create_transporter_prog(user_rb_fd, kernel_perf_fd);
	if (kernel_perf_fd >= 0)
		close(kernel_perf_fd);
	close(user_rb_fd);

	if (bpf_fd < 0) {
		if (pfd >= 0)
			close(pfd);
		throw std::runtime_error(
			"Unable to create transporter kernel ebpf program for shared perf event");
	}
	if (ioctl(pfd, PERF_EVENT_IOC_SET_BPF, bpf_fd) < 0) {
		close(bpf_fd);
		if (pfd >= 0)
			close(pfd);
		throw std::runtime_error("Failed to run PERF_EVENT_IOC_SET_BPF");
	}
	if (ioctl(pfd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
		close(bpf_fd);
		if (pfd >= 0)
			close(pfd);
		throw std::runtime_error("Failed to run PERF_EVENT_IOC_ENABLE");
	}
	try {
		local_key_ = ctx.allocate_local_key();
	} catch (...) {
		(void)ioctl(pfd, PERF_EVENT_IOC_DISABLE, 0);
		close(bpf_fd);
		if (pfd >= 0)
			close(pfd);
		throw;
	}
	auto *daemon_entry = g_perf_daemon_local_table.get_or_init(
		local_key_,
		[&](perf_daemon_local_entry &out) -> bool {
			out.pfd = pfd;
			out.transporter_prog_fd = bpf_fd;
			return true;
		},
		cleanup_perf_daemon_local);
	if (daemon_entry == nullptr) {
		(void)ioctl(pfd, PERF_EVENT_IOC_DISABLE, 0);
		close(bpf_fd);
		if (pfd >= 0)
			close(pfd);
		release_local_key(seg_alloc_.get_segment_manager(), local_key_);
		throw std::runtime_error(
			"Perf daemon local table slot unavailable");
	}
}

perf_event_array_kernel_user_impl::~perf_event_array_kernel_user_impl()
{
	g_perf_user_rb_local_table.evict(local_key_,
					 cleanup_perf_user_rb_local);
	g_perf_daemon_local_table.evict(local_key_, cleanup_perf_daemon_local);
	release_local_key(seg_alloc_.get_segment_manager(), local_key_);
}

void *perf_event_array_kernel_user_impl::elem_lookup(const void *key)
{
	int32_t k = *(int32_t *)key;
	if (k < 0 || (size_t)k >= max_ent) {
		errno = EINVAL;
		return nullptr;
	}
	SPDLOG_INFO(
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
	SPDLOG_INFO(
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
	SPDLOG_INFO(
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

// The original program was like:
/*
#define NULL 0
static long (*bpf_dynptr_read)(void *dst, int len, const struct bpf_dynptr *src,
int offset, long flags) = (void *) 201; static long
(*bpf_perf_event_output)(void *ctx, void *map, long flags, void *data, long
size) = (void *) 25; static long (*bpf_user_ringbuf_drain)(void *map, void
*callback_fn, void *ctx, long flags) = (void *) 209;


struct context {
    unsigned long size;
    char buf[32];
};
static int cb(void *dynptr, struct context *ctx);
int func(void* ctx)
{
    struct context ctx_buf;
    ctx_buf.size = 0;
    for(int i=0;i<sizeof(ctx_buf.buf);i++) ctx_buf.buf[i]=0;
	if(bpf_user_ringbuf_drain((void*)0x1234567812345678, &cb, &ctx_buf,
0)==1){ bpf_perf_event_output(ctx, (void*)0x1234567812345678, 0, ctx_buf.buf,
ctx_buf.size);
    }
    return 0;
}
static int cb(void *dynptr, struct context *ctx)
{
	if(bpf_dynptr_read(&ctx->size, 8, dynptr, 0, 0)<0) return 1;
	if(ctx->size>sizeof(ctx->buf)) return 1;
	if(bpf_dynptr_read(ctx->buf, ctx->size, dynptr, 8, 0)<0) return 1;
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
// r1 = map_by_fd(user_ringbuf_fd)
// BPF_LD_IMM64_RAW_FULL(1, 1, 0, 0, user_ringbuf_fd, 0),
#define ZERO(off) BPF_STX_MEM(BPF_DW, 10, 1, -off)
	bpf_insn insns[] = {
		// r6 = r1
		BPF_MOV64_REG(6, 1),
		// r1 = 0
		BPF_MOV64_IMM(1, 0),
		// r10-0x8...=r10-0x1c8 := 0
		ZERO(0x8), ZERO(0x10), ZERO(0x18), ZERO(0x20), ZERO(0x28),
		ZERO(0x30), ZERO(0x38), ZERO(0x40), ZERO(0x48), ZERO(0x50),
		ZERO(0x58), ZERO(0x60), ZERO(0x68), ZERO(0x70), ZERO(0x78),
		ZERO(0x80), ZERO(0x88), ZERO(0x90), ZERO(0x98), ZERO(0xa0),
		ZERO(0xa8), ZERO(0xb0), ZERO(0xb8), ZERO(0xc0), ZERO(0xc8),
		ZERO(0xd0), ZERO(0xd8), ZERO(0xe0), ZERO(0xe8), ZERO(0xf0),
		ZERO(0xf8), ZERO(0x100), ZERO(0x108), ZERO(0x110), ZERO(0x118),
		ZERO(0x120), ZERO(0x128), ZERO(0x130), ZERO(0x138), ZERO(0x140),
		ZERO(0x148), ZERO(0x150), ZERO(0x158), ZERO(0x160), ZERO(0x168),
		ZERO(0x170), ZERO(0x178), ZERO(0x180), ZERO(0x188), ZERO(0x190),
		ZERO(0x198), ZERO(0x1a0), ZERO(0x1a8), ZERO(0x1b0), ZERO(0x1b8),
		ZERO(0x1c0), ZERO(0x1c8),
		// r3 = r10
		BPF_MOV64_REG(3, 10),
		// r3 += -0x1e8
		BPF_ALU64_IMM(BPF_ADD, 3, -0x1c8),
		// r1 = map_by_fd(user_ringbuf_fd)
		BPF_LD_IMM64_RAW_FULL(1, 1, 0, 0, user_ringbuf_fd, 0),
		// r2 = code(callback)
		BPF_LD_IMM64_RAW_FULL(2, 4, 0, 0, 22, 0),
		// r4 = 0
		BPF_MOV64_IMM(4, 0),
		// call 0xd1 (bpf_user_ringbuf_drain)
		BPF_EMIT_CALL(0xd1),
		// r0 <<= 0x20
		BPF_ALU64_IMM(BPF_LSH, 0, 0x20),
		// r0 s>>= 0x20
		BPF_ALU64_IMM(BPF_ARSH, 0, 0x20),
		// r1 = 1
		BPF_MOV64_IMM(1, 1),
		// if r1 s> r0 goto +0xd
		BPF_RAW_INSN(0x6d, 1, 0, 0xd, 0),
		// r4 = r10
		BPF_MOV64_REG(4, 10),
		// r4 += -0x1e0
		BPF_ALU64_IMM(BPF_ADD, 4, -0x1c0),
		// r5 = *(u64 *)(r10 - 0x1e8)
		BPF_LDX_MEM(BPF_DW, 5, 10, -0x1c8),
		// r1 = 0x1e1
		BPF_MOV64_IMM(1, 0x1c1),
		// if r1 > r5 goto +0x2
		BPF_RAW_INSN(0x2d, 1, 5, 2, 0),
		// r5 = 0x1e0
		BPF_MOV64_IMM(5, 0x1c0),
		// *(u64 *)(r10 - 0x1e8) = r5
		BPF_STX_MEM(BPF_DW, 10, 5, -0x1c8),
		// r1 = r6
		BPF_MOV64_REG(1, 6),
		// r2 = map_by_fd(perf_event_array_fd)
		BPF_LD_IMM64_RAW_FULL(2, 1, 0, 0, perf_event_array_fd, 0),
		// r3 = 0xffffffff
		BPF_LD_IMM64_RAW_FULL(3, 0, 0, 0, (__s32)0xffffffff, 0),
		// call 0x19 bpf_perf_event_output
		BPF_EMIT_CALL(0x19),
		// r0 = 0
		BPF_MOV64_IMM(0, 0),
		// exit
		BPF_EXIT_INSN(),
		// callback function
		// r7 = r2
		BPF_MOV64_REG(7, 2),
		// r6 = r1
		BPF_MOV64_REG(6, 1),
		// r8 = 0
		BPF_MOV64_IMM(8, 0),
		// r1 = r7
		BPF_MOV64_REG(1, 7),
		// r2 = 8
		BPF_MOV64_IMM(2, 8),
		// r3 = r6
		BPF_MOV64_REG(3, 6),
		// r4 = 0
		BPF_MOV64_IMM(4, 0),
		// r5 = 0
		BPF_MOV64_IMM(5, 0),
		// call 0xc9 bpf_dynptr_read
		BPF_EMIT_CALL(0xc9),
		// if r8 s> r0 goto +0x8
		BPF_RAW_INSN(0x6d, 8, 0, 8, 0),
		// r2 = *(u64 *)(r7 + 0x0)
		BPF_LDX_MEM(BPF_DW, 2, 7, 0),
		// if r2 > 0x1e0 goto +0x6
		BPF_RAW_INSN(0x25, 2, 0, 6, 0x01c0),
		// r7 += 0x8
		BPF_ALU64_IMM(BPF_ADD, 7, 8),
		// r1 = r7
		BPF_MOV64_REG(1, 7),
		// r3 = r6
		BPF_MOV64_REG(3, 6),
		// r4 = 8
		BPF_MOV64_IMM(4, 8),
		// r5 = 0
		BPF_MOV64_IMM(5, 0),
		// call 0xc9
		BPF_EMIT_CALL(0xc9),
		// r0 = 1
		BPF_MOV64_IMM(0, 1),
		// exit
		BPF_EXIT_INSN()

	};
#undef ZERO
	return std::vector<uint64_t>((uint64_t *)&insns,
				     (uint64_t *)&insns + std::size(insns));
}

int create_intervally_triggered_perf_event(int duration_ms)
{
	perf_event_attr pe_attr;
	memset(&pe_attr, 0, sizeof(pe_attr));
	pe_attr.type = PERF_TYPE_SOFTWARE;
	pe_attr.size = sizeof(struct perf_event_attr);
	pe_attr.config = PERF_COUNT_SW_CPU_CLOCK;
	pe_attr.sample_period = (__u64)duration_ms * 1000;
	pe_attr.sample_type = PERF_SAMPLE_RAW;
	pe_attr.freq = 0;

#ifdef PERF_FLAG_FD_CLOEXEC
	const unsigned long perf_flags = PERF_FLAG_FD_CLOEXEC;
#else
	const unsigned long perf_flags = 0;
#endif
	return syscall(__NR_perf_event_open, &pe_attr, 0, -1, -1, perf_flags);
}
} // namespace bpftime

static int create_transporter_prog(int user_ringbuf_fd, int kernel_perf_fd)
{
	// Maniuplate a corresponding btf
	btf *btf = btf__new_empty();
	int bpf_dynptr_st = btf__add_struct(btf, "bpf_dynptr", 0);
	int bpf_dynptr_ptr = btf__add_ptr(btf, bpf_dynptr_st);
	int void_ptr = btf__add_ptr(btf, 0);
	int long_ty = btf__add_int(btf, "long", 8, 1);
	int int_ty = btf__add_int(btf, "int", 4, 1);

	int cb_func_proto = btf__add_func_proto(btf, long_ty);
	btf__add_func_param(btf, "dynptr", bpf_dynptr_ptr);
	btf__add_func_param(btf, "context", void_ptr);
	int cb_func = btf__add_func(btf, "transporter_cb", BTF_FUNC_STATIC,
				    cb_func_proto);
	// int main_func_proto = btf__add_func(btf,
	// "transporter",BTF_FUNC_GLOBAL , int proto_type_id)
	int main_func_proto = btf__add_func_proto(btf, int_ty);
	btf__add_func_param(btf, "ctx", void_ptr);
	int main_func = btf__add_func(btf, "transporter", BTF_FUNC_GLOBAL,
				      main_func_proto);
	uint32_t size;

	auto btf_raw_data = btf__raw_data(btf, &size);
	LIBBPF_OPTS(bpf_btf_load_opts, btf_load_opts);
	int btf_fd = bpf_btf_load(btf_raw_data, size, &btf_load_opts);
	if (btf_fd < 0) {
		SPDLOG_ERROR("Failed to load btf into kernel: {}", errno);
		return -1;
	}
	auto prog = bpftime::create_transporting_kernel_ebpf_program(
		user_ringbuf_fd, kernel_perf_fd);
	std::vector<int> fds;
	fds.push_back(user_ringbuf_fd);
	fds.push_back(kernel_perf_fd);
	std::vector<bpf_func_info> func_info;
	func_info.push_back(
		bpf_func_info{ .insn_off = 0, .type_id = (uint32_t)main_func });
	func_info.push_back(
		bpf_func_info{ .insn_off = 86, .type_id = (uint32_t)cb_func });

	LIBBPF_OPTS(bpf_prog_load_opts, prog_load_opts);
	// char log_buffer[8192];
	const size_t log_buffer_size = 1 << 20;
	char *log_buffer = new char[log_buffer_size];
	prog_load_opts.log_buf = log_buffer;
	prog_load_opts.log_size = log_buffer_size;
	prog_load_opts.log_level = 5;
	prog_load_opts.fd_array = fds.data();
	prog_load_opts.prog_btf_fd = btf_fd;
	prog_load_opts.func_info = func_info.data();
	prog_load_opts.func_info_cnt = func_info.size();
	prog_load_opts.func_info_rec_size = sizeof(bpf_func_info);
	SPDLOG_DEBUG("Loading transporter program with {} insns", prog.size());
	int bpf_fd = bpf_prog_load(BPF_PROG_TYPE_PERF_EVENT, "transporter",
				   "GPL", (bpf_insn *)prog.data(), prog.size(),
				   &prog_load_opts);
	std::string log_message(log_buffer);
	delete[] log_buffer;
	if (bpf_fd < 0) {
		SPDLOG_ERROR("Failed to load bpf prog: err={}, message=\n{}",
			     errno, log_message);
	}
	return bpf_fd;
}
