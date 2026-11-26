/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <bpf_map/userspace/ringbuf_map.hpp>
#include <pthread.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#if defined(__linux__)
#include <sys/eventfd.h>
#include <unistd.h>
#ifndef BPF_RB_NO_WAKEUP
#define BPF_RB_NO_WAKEUP 1U
#endif
#ifndef BPF_RB_FORCE_WAKEUP
#define BPF_RB_FORCE_WAKEUP 2U
#endif
#elif __APPLE__
#include "bpftime_epoll.h"
#endif

enum {
	BPF_RINGBUF_BUSY_BIT = 2147483648,
	BPF_RINGBUF_DISCARD_BIT = 1073741824,
	BPF_RINGBUF_HDR_SZ = 8,
};

static inline int roundup_len(__u32 len)
{
	len <<= 2;
	len >>= 2;
	len += BPF_RINGBUF_HDR_SZ;
	return (len + 7) / 8 * 8;
}

#define READ_ONCE_UL(x) (*(volatile unsigned long *)&x)
#define WRITE_ONCE_UL(x, v) (*(volatile unsigned long *)&x) = (v)
#define READ_ONCE_I(x) (*(volatile int *)&x)
#define WRITE_ONCE_I(x, v) (*(volatile int *)&x) = (v)

#if defined(__x86_64__)
#define barrier() asm volatile("" ::: "memory")
#define smp_store_release_ul(p, v)                                             \
	do {                                                                   \
		barrier();                                                     \
		WRITE_ONCE_UL(*p, v);                                          \
	} while (0)

#define smp_load_acquire_ul(p)                                                 \
	({                                                                     \
		unsigned long ___p = READ_ONCE_UL(*p);                         \
		barrier();                                                     \
		___p;                                                          \
	})

#define smp_load_acquire_i(p)                                                  \
	({                                                                     \
		int ___p = READ_ONCE_I(*p);                                    \
		barrier();                                                     \
		___p;                                                          \
	})

#elif defined(__aarch64__)
// https://github.com/torvalds/linux/blob/master/tools/arch/arm64/include/asm/barrier.h
#define smp_store_release_ul(p, v)                                             \
	do {                                                                   \
		asm volatile("stlr %1, %0" : "=Q"(*p) : "r"(v) : "memory");    \
	} while (0)
#define smp_load_acquire_ul(p)                                                 \
	({                                                                     \
		unsigned long ___p;                                            \
		asm volatile("ldar %0, %1" : "=r"(___p) : "Q"(*p) : "memory"); \
		___p;                                                          \
	})
#define smp_load_acquire_i(p)                                                  \
	({                                                                     \
		int ___p;                                                      \
		asm volatile("ldar %w0, %1"                                    \
			     : "=r"(___p)                                      \
			     : "Q"(*p)                                         \
			     : "memory");                                      \
		___p;                                                          \
	})
#elif defined(__riscv)
#define smp_store_release_ul(p, v)                                             \
	do {                                                                   \
		asm volatile("fence rw,w" ::: "memory");                       \
		WRITE_ONCE_UL(*p, v);                                          \
	} while (0)
#define smp_load_acquire_ul(p)                                                 \
	({                                                                     \
		unsigned long ___p = READ_ONCE_UL(*p);                         \
		asm volatile("fence r,rw" ::: "memory");                       \
		___p;                                                          \
	})
#define smp_load_acquire_i(p)                                                  \
	({                                                                     \
		int ___p = READ_ONCE_I(*p);                                    \
		asm volatile("fence r,rw" ::: "memory");                       \
		___p;                                                          \
	})
#else
#error Unsupported architecture
#endif

namespace bpftime
{

void *ringbuf_map_impl::elem_lookup(const void *key)
{
	SPDLOG_ERROR(
		"Trying to perform lookup on a ringbuf map, which is not supported");
	errno = ENOTSUP;
	return nullptr;
}

long ringbuf_map_impl::elem_update(const void *key, const void *value,
				   uint64_t flags)
{
	SPDLOG_ERROR(
		"Trying to perform update on a ringbuf map, which is not supported");
	errno = ENOTSUP;
	return -1;
}

long ringbuf_map_impl::elem_delete(const void *key)
{
	SPDLOG_ERROR(
		"Trying to perform delete on a ringbuf map, which is not supported");
	errno = ENOTSUP;
	return -1;
}

int ringbuf_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_ERROR(
		"Trying to perform map_get_next_key on a ringbuf map, which is not supported");
	errno = ENOTSUP;
	return -1;
}

ringbuf_map_impl::ringbuf_map_impl(
	uint32_t max_ent, boost::interprocess::managed_shared_memory &memory)
	: ringbuf_impl(boost::interprocess::make_managed_shared_ptr(
		  memory.construct<ringbuf>(
			  boost::interprocess::anonymous_instance)(max_ent,
								   memory),
		  memory))
{
}
ringbuf_weak_ptr ringbuf_map_impl::create_impl_weak_ptr()
{
	return ringbuf_weak_ptr(ringbuf_impl);
}
ringbuf_shared_ptr ringbuf_map_impl::create_impl_shared_ptr()
{
	return ringbuf_impl;
}
void *ringbuf_map_impl::get_consumer_page() const
{
	return ringbuf_impl->consumer_pos.get();
}
void *ringbuf_map_impl::get_producer_page() const
{
	return ringbuf_impl->producer_pos.get();
}

void *ringbuf_map_impl::reserve(size_t size, int self_fd)
{
	return ringbuf_impl->reserve(size, self_fd);
}
void ringbuf_map_impl::submit(const void *sample, bool discard)
{
	return ringbuf_impl->submit(sample, discard);
}

ringbuf::ringbuf(uint32_t max_ent,
		 boost::interprocess::managed_shared_memory &memory)
	: max_ent(max_ent),
	  raw_buffer(boost::interprocess::make_managed_unique_ptr(
		  memory.construct<buf_vec>(
			  boost::interprocess::anonymous_instance)(
			  getpagesize() * 2 + size_t(max_ent) * 2,
			  vec_allocator(memory.get_segment_manager())),
		  memory))
{
	const auto page_size = getpagesize();

	if ((size_t)page_size < sizeof(unsigned long)) {
		throw std::runtime_error(
			"Page size is expected to be greater than sizeof(unsigned long)");
	}
	consumer_pos = (unsigned long *)(uintptr_t)(&((*raw_buffer)[0]));
	producer_pos =
		(unsigned long *)(uintptr_t)(&((*raw_buffer)[page_size]));
	data = (uint8_t *)(uintptr_t)(&((*raw_buffer)[page_size * 2]));
#if defined(__linux__)
	// A non-blocking eventfd needs no capabilities: it is a plain kernel
	// object shared with consumers through SCM_RIGHTS, never through
	// /proc/<pid>/fd or pidfd_getfd.
	kernel_eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
	if (kernel_eventfd < 0) {
		SPDLOG_WARN(
			"Failed to create ringbuf eventfd, kernel epoll wakeup disabled: {}",
			errno);
		kernel_eventfd = -1;
	}
	eventfd_owner_pid = getpid();
#endif
}

int ringbuf::fetch_data(int (*cb)(void *ctx, void *data, size_t len), void *ctx)
{
	int *len_ptr, len, err;
	/* 64-bit to avoid overflow in case of extreme application behavior */
	int64_t cnt = 0;
	unsigned long cons_pos, prod_pos;
	bool got_new_data;
	void *sample;

	cons_pos = smp_load_acquire_ul(consumer_pos.get());
	do {
		got_new_data = false;
		prod_pos = smp_load_acquire_ul(producer_pos.get());
		while (cons_pos < prod_pos) {
			auto len_ptr =
				(int32_t *)(uintptr_t)(data.get() +
						       (cons_pos & mask()));
			len = smp_load_acquire_i(len_ptr);

			/* sample not committed yet, bail out for now */
			if (len & BPF_RINGBUF_BUSY_BIT)
				goto done;

			got_new_data = true;
			cons_pos += roundup_len(len);

			if ((len & BPF_RINGBUF_DISCARD_BIT) == 0) {
				sample = (void *)(((uintptr_t)len_ptr) +
						  BPF_RINGBUF_HDR_SZ);
				err = cb(ctx, sample, len);
				if (err < 0) {
					/* update consumer pos and bail out */
					smp_store_release_ul(consumer_pos.get(),
							     cons_pos);
					return err;
				}
				cnt++;
			}

			smp_store_release_ul(consumer_pos.get(), cons_pos);
		}
	} while (got_new_data);
done:
	return cnt;
}
bool ringbuf::has_data() const
{
	auto cons_pos = smp_load_acquire_ul(consumer_pos.get());
	auto prod_pos = smp_load_acquire_ul(producer_pos.get());
	if (cons_pos < prod_pos) {
		auto len_ptr = (int32_t *)(uintptr_t)(data.get() +
						      (cons_pos & mask()));
		auto len = smp_load_acquire_i(len_ptr);
		if ((len & BPF_RINGBUF_BUSY_BIT) == 0) {
			return true;
		}
	}
	return false;
}

struct ringbuf_hdr {
	uint32_t len;
	int32_t fd;
};

void *ringbuf::reserve(size_t size, int self_fd)
{
	if (size & (BPF_RINGBUF_BUSY_BIT | BPF_RINGBUF_DISCARD_BIT)) {
		errno = E2BIG;
		SPDLOG_ERROR(
			"Try to reserve an area of {} bytes, which is too big for ringbuf map {}",
			size, self_fd);
		return nullptr;
	}
	const unsigned long total_size =
		(size + BPF_RINGBUF_HDR_SZ + 7) / 8 * 8;
	if (total_size > max_ent) {
		errno = E2BIG;
		return nullptr;
	}

	// Publish BUSY_BIT headers before advancing producer_pos so a holder
	// killed mid-reserve leaves a valid ring for the next recovered owner.
	if (!reserve_lock.lock_and_recover()) {
		errno = EBUSY;
		return nullptr;
	}
	struct unlock_guard {
		const ringbuf &rb;
		~unlock_guard() { rb.reserve_lock.unlock(); }
	} guard{ *this };

	const unsigned long cons_pos =
		smp_load_acquire_ul(consumer_pos.get());
	const unsigned long prod_pos =
		smp_load_acquire_ul(producer_pos.get());
	const unsigned long off = prod_pos & mask();
	const bool need_padding = off + total_size > max_ent;
	const unsigned long record_pos =
		need_padding ? (prod_pos - off + max_ent) : prod_pos;
	if ((record_pos - cons_pos) + total_size > max_ent) {
		errno = ENOSPC;
		return nullptr;
	}

	uint32_t header_off = (uint32_t)off;
	if (need_padding) {
		auto pad = (ringbuf_hdr *)(uintptr_t)(data.get() + off);
		const uint32_t pad_len =
			(uint32_t)(max_ent - off - BPF_RINGBUF_HDR_SZ);
		pad->fd = self_fd;
		__atomic_store_n(&pad->len,
				 pad_len | BPF_RINGBUF_DISCARD_BIT,
				 __ATOMIC_RELEASE);
		header_off = 0;
	}
	auto header =
		(ringbuf_hdr *)(uintptr_t)(data.get() + header_off);
	header->fd = self_fd;
	__atomic_store_n(&header->len, (uint32_t)size | BPF_RINGBUF_BUSY_BIT,
			 __ATOMIC_RELEASE);
	smp_store_release_ul(producer_pos.get(), record_pos + total_size);

	auto ptr = data.get() + header_off + BPF_RINGBUF_HDR_SZ;
	SPDLOG_TRACE("ringbuf: reserved {} bytes at {}, fd {}", size,
		     (void *)ptr, self_fd);
	return ptr;
}

void ringbuf::submit(const void *sample, bool discard)
{
	uintptr_t hdr_offset = mask() + 1 + ((uint8_t *)sample - data.get()) -
			       BPF_RINGBUF_HDR_SZ;
	auto hdr =
		(ringbuf_hdr *)((uintptr_t)data.get() + (hdr_offset & mask()));

	auto new_len = hdr->len & ~BPF_RINGBUF_BUSY_BIT;
	if (discard)
		new_len |= BPF_RINGBUF_DISCARD_BIT;
	__atomic_exchange_n(&hdr->len, new_len, __ATOMIC_ACQ_REL);
}
ringbuf::~ringbuf()
{
#if defined(__linux__)
	// Only the creator owns the kernel fd; agents that received a dup via
	// SCM_RIGHTS close their own copy through their process-local cache.
	if (kernel_eventfd >= 0 && eventfd_owner_pid == getpid())
		close(kernel_eventfd);
#endif
}

bool ringbuf::should_wakeup(const void *sample, uint64_t flags) const
{
#if defined(__linux__)
	if (flags & BPF_RB_NO_WAKEUP)
		return false;
	if (flags & BPF_RB_FORCE_WAKEUP)
		return true;
	// Default adaptive behavior (mirrors libbpf ring_buffer__commit):
	// wake only when the consumer had caught up to the start of this
	// record, i.e. it is likely blocked waiting for new data. A consumer
	// still processing older records already has a level-triggered
	// eventfd ready. Positions are monotonic while the header lives at a
	// wrapped offset, so recover this record's absolute position from the
	// current producer position. Live data never exceeds the ring size,
	// making the reconstruction unique.
	unsigned long ring_sz = (unsigned long)mask() + 1;
	unsigned long off = (unsigned long)((const uint8_t *)sample -
					    data.get()) -
			     BPF_RINGBUF_HDR_SZ;
	unsigned long prod = smp_load_acquire_ul(producer_pos.get());
	// The record start is the unique lap-aligned position in
	// (prod - ring_sz, prod] whose offset matches the header slot.
	// Delta zero means the record spans the whole ring, so it starts one
	// full lap earlier rather than at prod.
	unsigned long delta = (prod - off) % ring_sz;
	if (delta == 0)
		delta = ring_sz;
	unsigned long record_start = prod - delta;
	return smp_load_acquire_ul(consumer_pos.get()) == record_start;
#else
	(void)sample;
	(void)flags;
	return false;
#endif
}
} // namespace bpftime
