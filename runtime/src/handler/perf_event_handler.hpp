#ifndef _PERF_EVENT_HANDLER
#define _PERF_EVENT_HANDLER
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>

namespace bpftime
{
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

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
	uint64_t offset;
	int pid;
	size_t ref_ctr_off;
	boost_shm_string _module_name;
	// Tracepoint id at /sys/kernel/tracing/events/syscalls/*/id, used to
	// indicate which syscall to trace
	int32_t tracepoint_id = -1;
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
};

} // namespace bpftime

#endif
