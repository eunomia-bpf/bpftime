#ifndef _WRAPPER_HPP
#define _WRAPPER_HPP

#include <vector>
#include <cinttypes>
#include <memory>
#include <ebpf-core.h>
#include "helpers_impl.hpp"
#include <unordered_map>
#include <deque>
#include <unordered_set>
#include <functional>
#include <linux/bpf.h>
#include <variant>
#include <atomic>
#include <pthread.h>
#include <string>

struct VectorHash {
	size_t operator()(const std::vector<uint8_t> &v) const
	{
		std::hash<uint8_t> hasher;
		size_t seed = 0;
		for (auto &i : v) {
			seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) +
				(seed >> 2);
		}
		return seed;
	}
};

struct EbpfProgWrapper {
	std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm;
	EbpfProgWrapper() : vm(ebpf_create(), ebpf_destroy)
	{
		inject_helpers(vm.get());
		ebpf_set_lddw_helpers(vm.get(), map_ptr_by_fd, nullptr, map_val,
				      nullptr, nullptr);
	}
};

using HashMapImpl = std::unordered_map<std::vector<uint8_t>,
				       std::vector<uint8_t>, VectorHash>;
using ArrayMapImpl = std::vector<std::vector<uint8_t> >;

/// Consuming implementation could be seen at
/// https://github.com/libbpf/libbpf/blob/05f94ddbb837f5f4b3161e341eed21be307eaa04/src/ringbuf.c#L227
/// libbpf will do the following mappings:
/// maps a whole page for storing `consumer_pos`, indicating the last record
/// that the consumer consumed maps a whole page for storing `producer_pos`,
/// indicating the last record that the procuder has produced In this way, once
/// consumer_pos < producer_pos, the consumer should consume records. maps
/// 2*max_ent bytes for holding data, for holding data that may across the
/// boundry of the ring. This mapping was concatancated with `producer_pos`

/// Positions will be masked by `max_ent-1`, to implement the "ring". So
/// `max_ent` must be a power of 2

/// What's in the ring?
/// A series of records. Each record was begun with 8bytes of header. The lower
/// 4 bytes was used to indicate the `length` of the data of this record. But
/// the 30bit of this u32 is used for BPF_RINGBUF_DISCARD_BIT, and the 31th bit
/// is used for BPF_RINGBUF_BUSY_BIT. The length was already rounded up to the
/// multiple of 8byte. After the 8bytes header was followed by the real data. If
/// the data was so long that it touches the boundry of the ring, it will
/// directly cross the boundry and lay its bytes after the boundry, that's why
/// we allocate 2*max_ent bytes. If bit BPF_RINGBUF_BUSY_BIT was set for len,
/// then we should not consume the data now. If BPF_RINGBUF_DISCARD_BIT was set,
/// we should not call the consumer callback, but should also step the
/// consumer_pos, just like a normal record.

/// The process of producing

struct RingBuffer {
	uint32_t max_ent;
	unsigned long *consumer_pos = nullptr;
	unsigned long *producer_pos = nullptr;
	uint8_t *data = nullptr;
	unsigned long mask;
	pthread_spinlock_t reserve_lock;

	void *reserve(size_t size, int self_fd);
	void submit(const void *buf, bool discard);
	bool has_data();
	// int consume(std::function<int(void *, void *, size_t)>);
	RingBuffer(uint32_t max_ent);
	~RingBuffer();
};

using RingBufMapImpl = std::shared_ptr<RingBuffer>;

struct EbpfMapWrapper {
	enum bpf_map_type type;
	uint32_t key_size;
	uint32_t value_size;
	uint32_t max_entries;
	uint64_t flags;
	std::string name;
	std::variant<HashMapImpl, ArrayMapImpl, RingBufMapImpl> impl;
	bool frozen = false;
	uint32_t ifindex = 0;
	uint32_t btf_vmlinux_value_type_id = 0;
	uint32_t netns_dev = 0;
	uint32_t netns_ino = 0;
	uint32_t btf_id = 0;
	uint32_t btf_key_type_id = 0;
	uint32_t btf_value_type_id = 0;
	uint64_t map_extra = 0;
	EbpfMapWrapper(const EbpfMapWrapper &) = delete;
	EbpfMapWrapper(EbpfMapWrapper &&) noexcept = default;
	EbpfMapWrapper &operator=(const EbpfMapWrapper &) = delete;
	EbpfMapWrapper &operator=(EbpfMapWrapper &&) noexcept = default;
	int mapDelete(const void *key);
	const void *mapLookup(const void *key);
	int mapUpdate(const void *key, const void *value, uint64_t flags);
	const void *first_value_addr();
	EbpfMapWrapper(enum bpf_map_type type, uint32_t key_size,
		       uint32_t value_size, uint32_t max_ent, uint64_t flags,
		       std::string name);
};
struct PerfEventWrapper {
	bool enabled = false;
};
struct BpfLinkWrapper {
	uint32_t prog_fd, target_fd;
	BpfLinkWrapper(uint32_t prog_fd, uint32_t target_fd)
		: prog_fd(prog_fd), target_fd(target_fd)
	{
	}
};
struct EpollWrapper {
	std::vector<std::weak_ptr<RingBuffer> > rbs;
};

using EbpfObj = std::variant<EbpfProgWrapper, EbpfMapWrapper, PerfEventWrapper,
			     BpfLinkWrapper, EpollWrapper>;

static const char *name_mapping[] = { "Program", "Map", "PerfEvent", "BpfLink",
				      "Epoll" };

extern std::unordered_map<int, std::unique_ptr<EbpfObj> > objs;
extern std::atomic_int next_fd;

#endif
