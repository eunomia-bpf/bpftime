#ifndef _BPFTIME_SOFTWARE_PERF_EVENT_DATA_HPP
#define _BPFTIME_SOFTWARE_PERF_EVENT_DATA_HPP

#include <cstdint>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "linux/perf_event.h"
#include <handlers/handler_common_def.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
namespace bpftime
{
namespace shm_common
{
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

} // namespace shm_common
} // namespace bpftime

#endif
