#include <handler/perf_event_handler.hpp>
namespace bpftime
{
// attach to replace or filter self define types
bpf_perf_event_handler::bpf_perf_event_handler(
	bpf_event_type type, uint64_t offset, int pid, const char *module_name,
	boost::interprocess::managed_shared_memory &mem)
	: type(type), offset(offset), pid(pid),
	  _module_name(char_allocator(mem.get_segment_manager()))
{
	this->_module_name = module_name;
}
// create uprobe/uretprobe with new perf event attr
bpf_perf_event_handler::bpf_perf_event_handler(
	bool is_retprobe, uint64_t offset, int pid, const char *module_name,
	size_t ref_ctr_off, boost::interprocess::managed_shared_memory &mem)
	: offset(offset), pid(pid), ref_ctr_off(ref_ctr_off),
	  _module_name(char_allocator(mem.get_segment_manager()))
{
	if (is_retprobe) {
		type = bpf_event_type::BPF_TYPE_URETPROBE;
	} else {
		type = bpf_event_type::BPF_TYPE_UPROBE;
	}
	this->_module_name = module_name;
}

// create tracepoint
bpf_perf_event_handler::bpf_perf_event_handler(
	int pid, int32_t tracepoint_id,
	boost::interprocess::managed_shared_memory &mem)
	: type(bpf_event_type::PERF_TYPE_TRACEPOINT), pid(pid),
	  _module_name(char_allocator(mem.get_segment_manager())),
	  tracepoint_id(tracepoint_id)
{
}
} // namespace bpftime
