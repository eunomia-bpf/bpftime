#include "link_handler.hpp"
#include "handler/perf_event_handler.hpp"
#include "spdlog/spdlog.h"
using namespace bpftime;

bpf_link_handler::bpf_link_handler(struct bpf_link_create_args args,
				   managed_shared_memory &mem)
	: args(args), prog_id(args.prog_fd), target_id(args.target_fd),
	  //   attach_target_ids(mem.get_segment_manager()),
	  link_attach_type(args.attach_type)
{
	if (args.attach_type == BPF_PERF_EVENT) {
		SPDLOG_DEBUG(
			"Initializing bpf link of type perf event, using link create args");
		if (args.perf_event.bpf_cookie != 0) {
			data = perf_event_link_data{
				.attach_cookie = args.perf_event.bpf_cookie
			};
			SPDLOG_DEBUG("Set attach cookie to {}",
				     args.perf_event.bpf_cookie);
		} else {
			data = perf_event_link_data{};
		}
		// attach_target_ids.push_back(args.target_fd);
	} else if (args.attach_type == BPF_TRACE_UPROBE_MULTI) {
		SPDLOG_DEBUG(
			"Initializing bpf link of type uprobe_multi, using link create args");
		const auto &opts = args.uprobe_multi;
		uprobe_multi_entry_vector entries(uprobe_multi_entry_allocator(
			mem.get_segment_manager()));
		for (size_t i = 0; i < opts.cnt; i++) {
			entries.push_back(uprobe_multi_entry{
				.offset = ((unsigned long *)(uintptr_t)
						   opts.offsets)[i],
				.ref_ctr_offset =
					opts.ref_ctr_offsets != 0 ?
						((unsigned long *)(uintptr_t)opts
							 .ref_ctr_offsets)[i] :
						0,
				.cookie = (opts.cookies == 0 ||
					   ((uint64_t *)(uintptr_t)
						    opts.cookies)[i] == 0) ?
						  0 :
						  ((uint64_t *)(uintptr_t)
							   opts.cookies)[i] });
		}
		data = uprobe_multi_link_data{
			.path = boost_shm_string(
				(const char *)(uintptr_t)opts.path,
				char_allocator(mem.get_segment_manager())),
			.entries = entries,
			.flags = opts.flags,
			.pid = static_cast<int>(opts.pid)
		};
	} else {
		SPDLOG_ERROR("Unsupport bpf_link attach type: {}",
			     link_attach_type);
		throw std::runtime_error("Unsupported bpf_link attach type");
	}
}
bpf_link_handler::bpf_link_handler(int prog_id, int attach_target_id,
				   managed_shared_memory &mem)
	: prog_id(prog_id), target_id(attach_target_id),
	  link_attach_type(BPF_PERF_EVENT),
	  data(perf_event_link_data{ .attach_cookie = {} })
{
	// attach_target_ids.push_back(attach_target_id);
}
bpf_link_handler::bpf_link_handler(int prog_id, int attach_target_id,
				   std::optional<uint64_t> cookie,
				   managed_shared_memory &mem)
	: prog_id(prog_id), target_id(attach_target_id),
	  link_attach_type(BPF_PERF_EVENT),
	  data(perf_event_link_data{ .attach_cookie = cookie })
{
	// attach_target_ids.push_back(attach_target_id);
}
