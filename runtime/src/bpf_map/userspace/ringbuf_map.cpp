/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <bpf_map/userspace/ringbuf_map.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#if __APPLE__
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
#else
#error Only supports x86_64 and aarch64
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
	  reserve_mutex(boost::interprocess::make_managed_unique_ptr(
		  memory.construct<
			  boost::interprocess::interprocess_sharable_mutex>(
			  boost::interprocess::anonymous_instance)(),
		  memory)),
	  raw_buffer(boost::interprocess::make_managed_unique_ptr(
		  memory.construct<buf_vec>(
			  boost::interprocess::anonymous_instance)(
			  getpagesize() * 2 + max_ent * 2,
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
}

int ringbuf::fetch_data(std::function<int(void *, int)> cb)
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
				err = cb(sample, len);
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
using boost::interprocess::interprocess_sharable_mutex;
using boost::interprocess::sharable_lock;

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
	sharable_lock<interprocess_sharable_mutex> guard(*reserve_mutex);
	auto cons_pos = smp_load_acquire_ul(consumer_pos.get());
	auto prod_pos = smp_load_acquire_ul(producer_pos.get());
	auto avail_size = max_ent - (prod_pos - cons_pos);
	auto total_size = (size + BPF_RINGBUF_HDR_SZ + 7) / 8 * 8;
	if (total_size > max_ent) {
		errno = E2BIG;
		return nullptr;
	}
	if (avail_size < total_size) {
		errno = ENOSPC;
		return nullptr;
	}
	auto header =
		(ringbuf_hdr *)((uintptr_t)data.get() + (prod_pos & mask()));
	header->len = size | BPF_RINGBUF_BUSY_BIT;
	header->fd = self_fd;
	smp_store_release_ul(producer_pos.get(), prod_pos + total_size);
	auto ptr = data.get() + ((prod_pos + BPF_RINGBUF_HDR_SZ) & mask());
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

} // namespace bpftime
