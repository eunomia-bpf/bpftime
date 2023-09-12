#include <asm-generic/errno-base.h>
#include <linux/bpf.h>
#include <cinttypes>
#include "wrapper.hpp"
#include "ringbuf_helpers.h"
#include <cassert>
#include <iostream>
#include <mutex>
#include <atomic>
#include <ostream>
#include <pthread.h>

/* 8-byte ring buffer header structure */
struct ringbuf_hdr {
	uint32_t len;
	int32_t fd;
};

std::unordered_map<int, std::unique_ptr<EbpfObj> > objs;
std::atomic_int next_fd(1 << 20);

static inline int roundup_len(uint32_t len)
{
	/* clear out top 2 bits (discard and busy, if set) */
	len <<= 2;
	len >>= 2;
	/* add length prefix */
	len += BPF_RINGBUF_HDR_SZ;
	/* round up to 8 byte alignment */
	return (len + 7) / 8 * 8;
}

bool RingBuffer::has_data()
{
	auto cons_pos = smp_load_acquire_ul(consumer_pos);
	auto prod_pos = smp_load_acquire_ul(producer_pos);
	if (cons_pos < prod_pos) {
		auto len_ptr = (int32_t *)(uintptr_t)(data + (cons_pos & mask));
		auto len = smp_load_acquire_i(len_ptr);
		if ((len & BPF_RINGBUF_BUSY_BIT) == 0) {
			return true;
		}
	}
	return false;
}

RingBuffer::~RingBuffer()
{
	pthread_spin_destroy(&reserve_lock);
}
RingBuffer::RingBuffer(uint32_t max_ent) : max_ent(max_ent)
{
	this->mask = max_ent - 1;
	pthread_spin_init(&reserve_lock, 0);
}
void *RingBuffer::reserve(size_t size, int self_fd)
{
	if (size & (BPF_RINGBUF_BUSY_BIT | BPF_RINGBUF_DISCARD_BIT)) {
		errno = E2BIG;
		return nullptr;
	}
	pthread_spin_lock(&reserve_lock);
	auto cons_pos = smp_load_acquire_ul(consumer_pos);
	auto prod_pos = smp_load_acquire_ul(producer_pos);
	auto avail_size = max_ent - (prod_pos - cons_pos);
	auto total_size = (size + BPF_RINGBUF_HDR_SZ + 7) / 8 * 8;
	if (total_size > max_ent) {
		errno = E2BIG;
		pthread_spin_unlock(&reserve_lock);
		return nullptr;
	}
	if (avail_size < total_size) {
		errno = ENOSPC;
		pthread_spin_unlock(&reserve_lock);
		return nullptr;
	}
	auto header = (ringbuf_hdr *)((uintptr_t)data + (prod_pos & mask));
	header->len = size | BPF_RINGBUF_BUSY_BIT;
	header->fd = self_fd;
	smp_store_release_ul(producer_pos, prod_pos + total_size);
	pthread_spin_unlock(&reserve_lock);
	return data + ((prod_pos + BPF_RINGBUF_HDR_SZ) & mask);
}
void RingBuffer::submit(const void *sample, bool discard)
{
	uintptr_t hdr_offset =
		mask + 1 + ((uint8_t *)sample - data) - BPF_RINGBUF_HDR_SZ;
	auto hdr = (ringbuf_hdr *)((uintptr_t)data + (hdr_offset & mask));

	auto new_len = hdr->len & ~BPF_RINGBUF_BUSY_BIT;
	if (discard)
		new_len |= BPF_RINGBUF_DISCARD_BIT;
	__atomic_exchange_n(&hdr->len, new_len, __ATOMIC_ACQ_REL);
}

int EbpfMapWrapper::mapDelete(const void *key)
{
	if (type == BPF_MAP_TYPE_HASH) {
		HashMapImpl &map = std::get<HashMapImpl>(impl);
		std::vector<uint8_t> key_vec((uint8_t *)key,
					     (uint8_t *)key + key_size);
		map.erase(key_vec);
		return 0;
	} else if (type == BPF_MAP_TYPE_ARRAY) {
		return -ENOTSUP;
	} else {
		assert(false);
	}
}
const void *EbpfMapWrapper::mapLookup(const void *key)
{
	if (type == BPF_MAP_TYPE_HASH) {
		HashMapImpl &map = std::get<HashMapImpl>(impl);
		std::vector<uint8_t> key_vec((uint8_t *)key,
					     (uint8_t *)key + key_size);
		if (auto itr = map.find(key_vec); itr != map.end()) {
			return (const void *)&itr->second[0];
		} else {
			return nullptr;
		}

	} else if (type == BPF_MAP_TYPE_ARRAY) {
		auto key_val = *(uint32_t *)key;

		ArrayMapImpl &map = std::get<ArrayMapImpl>(impl);
		if (key_val >= map.size())
			return nullptr;
		return (const void *)&map.at(key_val)[0];
	} else {
		assert(false);
	}
}

int EbpfMapWrapper::mapUpdate(const void *key, const void *value,
			      uint64_t flags)
{
	if (frozen) {
		return -EINVAL;
	}
	if (type == BPF_MAP_TYPE_HASH) {
		HashMapImpl &map = std::get<HashMapImpl>(impl);
		std::vector<uint8_t> key_vec((uint8_t *)key,
					     (uint8_t *)key + key_size);
		std::vector<uint8_t> value_vec((uint8_t *)value,
					       (uint8_t *)value + value_size);
		map[key_vec] = value_vec;

	} else if (type == BPF_MAP_TYPE_ARRAY) {
		auto key_val = *(uint32_t *)key;

		std::vector<uint8_t> value_vec((uint8_t *)value,
					       (uint8_t *)value + value_size);
		ArrayMapImpl &map = std::get<ArrayMapImpl>(impl);
		if (key_val >= map.size())
			return -E2BIG;
		map[key_val] = value_vec;
	}
	return 0;
}

const void *EbpfMapWrapper::first_value_addr()
{
	if (type == BPF_MAP_TYPE_HASH) {
		HashMapImpl &map = std::get<HashMapImpl>(impl);
		if (map.empty())
			return nullptr;
		return (const void *)&map.begin()->second[0];

	} else if (type == BPF_MAP_TYPE_ARRAY) {
		ArrayMapImpl &map = std::get<ArrayMapImpl>(impl);
		return (const void *)&map.at(0)[0];
	} else {
		assert(false);
	}
}

EbpfMapWrapper::EbpfMapWrapper(enum bpf_map_type type, uint32_t key_size,
			       uint32_t value_size, uint32_t max_ent,
			       uint64_t flags, std::string name)
	: type(type), key_size(key_size), value_size(value_size),
	  max_entries(max_ent), flags(flags)

{
	switch (type) {
	case BPF_MAP_TYPE_HASH: {
		impl = std::unordered_map<std::vector<uint8_t>,
					  std::vector<uint8_t>, VectorHash>();
		break;
	}
	case BPF_MAP_TYPE_ARRAY: {
		impl = std::vector<std::vector<uint8_t> >(
			max_entries, std::vector<uint8_t>(value_size, 0));
		break;
	}
	case BPF_MAP_TYPE_RINGBUF: {
		if (max_ent == 0) {
			std::cerr << "Unexpected max entries: " << max_ent
				  << std::endl;
			exit(1);
		}
		uint32_t new_max_ent = 1;
		while (new_max_ent < max_ent)
			new_max_ent <<= 1;
		max_ent = new_max_ent;
		// Round up to the power of 2
		std::cerr << "Creating ringbuf map `" << name << "`"
			  << std::endl;
		impl = std::make_shared<RingBuffer>(max_ent);
		break;
	}
	default: {
		std::cerr << "Unsupported map type " << type << std::endl;
		exit(1);
	}
	}
}
