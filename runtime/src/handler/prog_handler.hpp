#ifndef _PROG_HANDLER_HPP
#define _PROG_HANDLER_HPP
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <ebpf-vm.h>
namespace bpftime
{

using managed_shared_memory = boost::interprocess::managed_shared_memory;
using char_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using boost_shm_string =
	boost::interprocess::basic_string<char, std::char_traits<char>,
					  char_allocator>;

// bpf progs handler
// in share memory. This is only a simple data struct to store the
// bpf program data.
class bpf_prog_handler {
    public:
	/* Note that tracing related programs such as
	 * BPF_PROG_TYPE_{KPROBE,TRACEPOINT,PERF_EVENT,RAW_TRACEPOINT}
	 * are not subject to a stable API since kernel internal data
	 * structures can change from release to release and may
	 * therefore break existing tracing BPF programs. Tracing BPF
	 * programs correspond to /a/ specific kernel which is to be
	 * analyzed, and not /a/ specific kernel /and/ all future ones.
	 */
	enum class bpf_prog_type {
		BPF_PROG_TYPE_UNSPEC,
		BPF_PROG_TYPE_SOCKET_FILTER,
		BPF_PROG_TYPE_KPROBE,
		BPF_PROG_TYPE_SCHED_CLS,
		BPF_PROG_TYPE_SCHED_ACT,
		BPF_PROG_TYPE_TRACEPOINT,
		BPF_PROG_TYPE_XDP,
		BPF_PROG_TYPE_PERF_EVENT,
		BPF_PROG_TYPE_CGROUP_SKB,
		BPF_PROG_TYPE_CGROUP_SOCK,
		BPF_PROG_TYPE_LWT_IN,
		BPF_PROG_TYPE_LWT_OUT,
		BPF_PROG_TYPE_LWT_XMIT,
		BPF_PROG_TYPE_SOCK_OPS,
		BPF_PROG_TYPE_SK_SKB,
		BPF_PROG_TYPE_CGROUP_DEVICE,
		BPF_PROG_TYPE_SK_MSG,
		BPF_PROG_TYPE_RAW_TRACEPOINT,
		BPF_PROG_TYPE_CGROUP_SOCK_ADDR,
		BPF_PROG_TYPE_LWT_SEG6LOCAL,
		BPF_PROG_TYPE_LIRC_MODE2,
		BPF_PROG_TYPE_SK_REUSEPORT,
		BPF_PROG_TYPE_FLOW_DISSECTOR,
		BPF_PROG_TYPE_CGROUP_SYSCTL,
		BPF_PROG_TYPE_RAW_TRACEPOINT_WRITABLE,
		BPF_PROG_TYPE_CGROUP_SOCKOPT,
		BPF_PROG_TYPE_TRACING,
		BPF_PROG_TYPE_STRUCT_OPS,
		BPF_PROG_TYPE_EXT,
		BPF_PROG_TYPE_LSM,
		BPF_PROG_TYPE_SK_LOOKUP,
		BPF_PROG_TYPE_SYSCALL, /* a program that can execute syscalls */
		BPF_PROG_TYPE_NETFILTER,
	};
	enum bpf_prog_type type;

	bpf_prog_handler(managed_shared_memory &mem,
			 const struct ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type);
	bpf_prog_handler(const bpf_prog_handler &) = delete;
	bpf_prog_handler(bpf_prog_handler &&) noexcept = default;
	bpf_prog_handler &operator=(const bpf_prog_handler &) = delete;
	bpf_prog_handler &operator=(bpf_prog_handler &&) noexcept = default;

	using shm_ebpf_inst_vector_allocator = boost::interprocess::allocator<
		ebpf_inst, managed_shared_memory::segment_manager>;

	using inst_vector =
		boost::interprocess::vector<ebpf_inst,
					    shm_ebpf_inst_vector_allocator>;
	inst_vector insns;

	using shm_int_vector_allocator = boost::interprocess::allocator<
		int, managed_shared_memory::segment_manager>;

	using attach_fds_vector =
		boost::interprocess::vector<int, shm_int_vector_allocator>;
	mutable attach_fds_vector attach_fds;

	void add_attach_fd(int fd) const
	{
		attach_fds.push_back(fd);
	}

	boost_shm_string name;
};

} // namespace bpftime

#endif
