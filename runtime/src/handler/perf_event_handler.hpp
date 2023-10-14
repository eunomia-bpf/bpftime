#ifndef _PERF_EVENT_HANDLER
#define _PERF_EVENT_HANDLER
#include "spdlog/spdlog.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <cstddef>
#include <optional>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
#include "linux/perf_event.h"
namespace bpftime
{

using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

struct perf_sample_raw {
	struct perf_event_header header;
	uint32_t size;
	char data[];
};

struct perf_sample_lost {
	struct perf_event_header header;
	uint64_t id;
	uint64_t lost;
	uint64_t sample_id;
};

/*
Implementation on the perf event ring buffer

There are two pointers, data_head and data_tail. Where data_head indicates the
position the next output should be laid at, and data_tail points to the position
the next input should read

	    data_tail               data_head
	    ^                       ^
	    |                       |
+-------+-------+-------+-------+--------+
| empty | data1 | data2 | data3 | unused |
+-------+-------+-------+-------+--------+
0                                        buf_len
When the emitter (the side that produce data) wants to output something:
- check if data_head meets data_tail under the modular of buffer length
- If not reached, put an instance of perf_sample_raw at data_head, fill it with
corresponding data, then fill the data to output. Note that the data may be cut
into two pieces, one of which will be laid at the tail, and another will be laid
at the head, if the remaining buffer space at the tail is not enough
- Add data_head with the corresponding size. modular with buf_len
*/

struct software_perf_event_data {
	using bytes_vec_allocator = boost::interprocess::allocator<
		uint8_t,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using bytes_vec =
		boost::interprocess::vector<uint8_t, bytes_vec_allocator>;
	int cpu;
	// Field `config` of perf_event_attr
	int64_t config;
	// Field `sample_type` of perf_event_attr
	int32_t sample_type;
	int pagesize;
	bytes_vec mmap_buffer;
	bytes_vec copy_buffer;
	software_perf_event_data(
		int cpu, int64_t config, int32_t sample_type,
		boost::interprocess::managed_shared_memory &memory);
	void *ensure_mmap_buffer(size_t buffer_size);
	perf_event_mmap_page &get_header_ref();
	const perf_event_mmap_page &get_header_ref_const() const;
	int output_data(const void *buf, size_t size);
	size_t mmap_size() const;
	bool has_data() const;
};

using software_perf_event_shared_ptr = boost::interprocess::managed_shared_ptr<
	software_perf_event_data,
	boost::interprocess::managed_shared_memory::segment_manager>::type;
using software_perf_event_weak_ptr = boost::interprocess::managed_weak_ptr<
	software_perf_event_data,
	boost::interprocess::managed_shared_memory::segment_manager>::type;

// perf event handler
struct bpf_perf_event_handler {
	enum class bpf_event_type {
		PERF_TYPE_HARDWARE = 0,
		PERF_TYPE_SOFTWARE = 1,
		PERF_TYPE_TRACEPOINT = 2,
		PERF_TYPE_HW_CACHE = 3,
		PERF_TYPE_RAW = 4,
		PERF_TYPE_BREAKPOINT = 5,

		// custom types
		BPF_TYPE_UPROBE = 6,
		BPF_TYPE_URETPROBE = 7,
		BPF_TYPE_FILTER = 8,
		BPF_TYPE_REPLACE = 9,
	} type;
	int enable() const
	{
		// TODO: implement enable logic.
		// If This is a server, should inject the agent into the target
		// process.
		return 0;
	}
	int disable() const
	{
		spdlog::debug("Disabling perf event, but nothing todo");
		return 0;
	}
	uint64_t offset;
	int pid;
	size_t ref_ctr_off;
	boost_shm_string _module_name;
	// Tracepoint id at /sys/kernel/tracing/events/syscalls/*/id, used to
	// indicate which syscall to trace
	int32_t tracepoint_id = -1;

	// Things needed by software perf event
	std::optional<software_perf_event_shared_ptr> sw_perf;

	std::optional<software_perf_event_weak_ptr>
	try_get_software_perf_data_weak_ptr() const;
	std::optional<void *>
	try_get_software_perf_data_raw_buffer(size_t buffer_size) const;
	// attach to replace or filter self define types
	bpf_perf_event_handler(bpf_event_type type, uint64_t offset, int pid,
			       const char *module_name,
			       boost::interprocess::managed_shared_memory &mem);
	// create uprobe/uretprobe with new perf event attr
	bpf_perf_event_handler(bool is_retprobe, uint64_t offset, int pid,
			       const char *module_name, size_t ref_ctr_off,
			       boost::interprocess::managed_shared_memory &mem);

	// create tracepoint
	bpf_perf_event_handler(int pid, int32_t tracepoint_id,
			       boost::interprocess::managed_shared_memory &mem);
	// create software perf event
	bpf_perf_event_handler(int cpu, int32_t sample_type, int64_t config,
			       boost::interprocess::managed_shared_memory &mem);
};

} // namespace bpftime

#endif
