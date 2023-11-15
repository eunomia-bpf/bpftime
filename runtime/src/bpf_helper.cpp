#include "bpftime_helper_group.hpp"
#include <cassert>
#include <cerrno>
#include <sched.h>
#ifdef ENABLE_BPFTIME_VERIFIER
#include "bpftime-verifier.hpp"
#endif
#include "spdlog/spdlog.h"
#include <map>
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <time.h>
#include <unistd.h>
#include <ctime>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_internal.h"
#include <spdlog/spdlog.h>
#include <vector>

#define PATH_MAX 4096

using namespace std;

extern "C" {

uint64_t bpftime_get_func_arg(uint64_t ctx, uint32_t n, uint64_t *value);

uint64_t bpftime_get_func_ret(uint64_t ctx, uint64_t *value);

// long bpftime_get_func_arg_cnt(void *ctx);

uint64_t bpftime_get_retval(void);

uint64_t bpftime_set_retval(uint64_t retval);

uint64_t bpftime_trace_printk(uint64_t fmt, uint64_t fmt_size, ...)
{
	const char *fmt_str = (const char *)fmt;
	va_list args;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wvarargs"
	va_start(args, fmt_str);
	long ret = vprintf(fmt_str, args);
#pragma GCC diagnostic pop
	va_end(args);
	return 0;
}

long bpftime_strncmp(const char *s1, uint64_t s1_sz, const char *s2)
{
	return strncmp(s1, s2, s1_sz);
}

uint64_t bpftime_probe_read(uint64_t dst, uint64_t size, uint64_t ptr, uint64_t,
			    uint64_t)
{
	memcpy((void *)(uintptr_t)dst, (void *)(uintptr_t)ptr,
	       (size_t)(uint32_t)(size));
	return 0;
}

uint64_t bpftime_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				     uint64_t)
{
	timespec spec;
	clock_gettime(CLOCK_MONOTONIC_COARSE, &spec);
	return spec.tv_sec * (uint64_t)1000000000 + spec.tv_nsec;
}

uint64_t bpftime_get_current_pid_tgid(uint64_t, uint64_t, uint64_t, uint64_t,
				      uint64_t)
{
	static int pid = getpid();
	return ((uint64_t)pid << 32) | pid;
}

uint64_t bpf_get_current_uid_gid(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t)
{
	static int gid = getgid();
	return (((uint64_t)gid) << 32) | gid;
}

uint64_t bpftime_ktime_get_ns(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	timespec spec;
	clock_gettime(CLOCK_MONOTONIC, &spec);
	return spec.tv_sec * (uint64_t)1000000000 + spec.tv_nsec;
}

uint64_t bpftime_get_current_comm(uint64_t buf, uint64_t size, uint64_t,
				  uint64_t, uint64_t)
{
	static std::string filename_buf;

	if (filename_buf.empty()) {
		char strbuf[PATH_MAX];

		auto len = readlink("/proc/self/exe", strbuf,
				    std::size(strbuf) - 1);
		if (len == -1)
			return 1;
		strbuf[len] = 0;
		auto str = std::string_view(strbuf);
		auto last_slash = str.find_last_of('/');

		auto filename = std::string(str.substr(last_slash + 1));
		filename_buf = filename;
	}
	strncpy((char *)(uintptr_t)buf, filename_buf.c_str(), (size_t)size);
	return 0;
}

uint64_t bpftime_map_lookup_elem_helper(uint64_t map, uint64_t key, uint64_t,
					uint64_t, uint64_t)
{
	return (uint64_t)bpftime_helper_map_lookup_elem(map >> 32, (void *)key);
}

uint64_t bpftime_map_update_elem_helper(uint64_t map, uint64_t key,
					uint64_t value, uint64_t flags,
					uint64_t)
{
	return (uint64_t)bpftime_helper_map_update_elem(map >> 32, (void *)key,
							(void *)value, flags);
}

uint64_t bpftime_map_delete_elem_helper(uint64_t map, uint64_t key, uint64_t,
					uint64_t, uint64_t)
{
	return (uint64_t)bpftime_helper_map_delete_elem(map >> 32, (void *)key);
}

uint64_t bpf_probe_read_str(uint64_t buf, uint64_t bufsz, uint64_t ptr,
			    uint64_t, uint64_t)
{
	strncpy((char *)(uintptr_t)buf, (const char *)(uintptr_t)ptr,
		(size_t)bufsz);
	return 0;
}

uint64_t bpf_get_stack(uint64_t, uint64_t buf, uint64_t sz, uint64_t, uint64_t)
{
	// TODO: implement this
	memset((void *)(uintptr_t)buf, 0, sz);
	return sz;
}

uint64_t bpf_ktime_get_coarse_ns(uint64_t, uint64_t, uint64_t, uint64_t,
				 uint64_t)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
	return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

uint64_t bpf_ringbuf_output(uint64_t rb, uint64_t data, uint64_t size,
			    uint64_t flags, uint64_t)
{
	int fd = (int)(rb >> 32);
	if (flags != 0) {
		spdlog::warn(
			"Currently only supports ringbuf_output with flags=0");
	}
	auto buf = bpftime_ringbuf_reserve(fd, size);
	if (!buf) {
		spdlog::error(
			"Failed to reserve when executing ringbuf output");
		return (uint64_t)-1;
	}
	memcpy(buf, (const void *)(uintptr_t)data, size);
	bpftime_ringbuf_submit(fd, buf, false);
	return 0;
}

uint64_t bpf_ringbuf_reserve(uint64_t rb, uint64_t size, uint64_t flags,
			     uint64_t, uint64_t)
{
	int fd = (int)(rb >> 32);
	if (flags != 0) {
		spdlog::warn(
			"Currently only supports ringbuf_reserve with flags=0");
	}
	return (uint64_t)(uintptr_t)bpftime_ringbuf_reserve(fd, size);
}

uint64_t bpf_ringbuf_submit(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			    uint64_t)
{
	int32_t *ptr = (int32_t *)(uintptr_t)data;
	int fd = ptr[-1];
	if (flags != 0) {
		spdlog::warn(
			"Currently only supports ringbuf_submit with flags=0");
	}
	bpftime_ringbuf_submit(fd, (void *)(uintptr_t)data, false);
	return 0;
}
uint64_t bpf_ringbuf_discard(uint64_t data, uint64_t flags, uint64_t, uint64_t,
			     uint64_t)
{
	int32_t *ptr = (int32_t *)(uintptr_t)data;
	int fd = ptr[-1];
	if (flags != 0) {
		spdlog::warn(
			"Currently only supports ringbuf_submit with flags=0");
	}
	bpftime_ringbuf_submit(fd, (void *)(uintptr_t)data, true);
	return 0;
}

uint64_t bpf_perf_event_output(uint64_t ctx, uint64_t map, uint64_t flags,
			       uint64_t data, uint64_t size)
{
	int32_t current_cpu = sched_getcpu();
	assert(current_cpu != -1);
	cpu_set_t mask, orig;
	CPU_ZERO(&mask);
	CPU_SET(current_cpu, &mask);
	sched_getaffinity(0, sizeof(orig), &orig);
	// Bind to the current cpu
	if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
		spdlog::error("Failed to set cpu affinity: {}", errno);
		errno = EINVAL;
		return (uint64_t)(-1);
	}
	int fd = map >> 32;
	// Check map type. userspace perf event array, or shared perf event
	// array?
	bpftime::bpf_map_type map_ty;
	if (int err = bpftime_map_get_info(fd, nullptr, nullptr, &map_ty);
	    err < 0) {
		spdlog::error("Unable to query map type of fd {}", fd);
		return -1;
	}
	int ret;
	if (map_ty == bpftime::bpf_map_type::BPF_MAP_TYPE_PERF_EVENT_ARRAY) {
		const int32_t *val_ptr =
			(int32_t *)(uintptr_t)bpftime_helper_map_lookup_elem(
				fd, &current_cpu);
		if (val_ptr == nullptr) {
			spdlog::error(
				"Invalid map fd for perf event output: {}", fd);
			errno = EINVAL;
			return (uint64_t)(-1);
		}
		int32_t perf_handler_fd = *val_ptr;
		ret = bpftime_perf_event_output(perf_handler_fd,
						(const void *)(uintptr_t)data,
						(size_t)size);
	} else if (map_ty ==
		   bpftime::bpf_map_type::
			   BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY) {
		ret = bpftime_shared_perf_event_output(
			fd, (const void *)(uintptr_t)data, (size_t)size);
	} else {
		spdlog::error(
			"Attempting to run perf_output on a non-perf array map");
		ret = -1;
	}

	sched_setaffinity(0, sizeof(orig), &orig);
	return (uint64_t)ret;
}
} // extern "C"

namespace bpftime
{

// copied from kernel/include/uapi/linux/bpf.h
enum bpf_func_id {
	BPF_FUNC_unspec = 0,
	BPF_FUNC_map_lookup_elem = 1,
	BPF_FUNC_map_update_elem = 2,
	BPF_FUNC_map_delete_elem = 3,
	BPF_FUNC_probe_read = 4,
	BPF_FUNC_ktime_get_ns = 5,
	BPF_FUNC_trace_printk = 6,
	BPF_FUNC_get_prandom_u32 = 7,
	BPF_FUNC_get_smp_processor_id = 8,
	BPF_FUNC_skb_store_bytes = 9,
	BPF_FUNC_l3_csum_replace = 10,
	BPF_FUNC_l4_csum_replace = 11,
	BPF_FUNC_tail_call = 12,
	BPF_FUNC_clone_redirect = 13,
	BPF_FUNC_get_current_pid_tgid = 14,
	BPF_FUNC_get_current_uid_gid = 15,
	BPF_FUNC_get_current_comm = 16,
	BPF_FUNC_get_cgroup_classid = 17,
	BPF_FUNC_skb_vlan_push = 18,
	BPF_FUNC_skb_vlan_pop = 19,
	BPF_FUNC_skb_get_tunnel_key = 20,
	BPF_FUNC_skb_set_tunnel_key = 21,
	BPF_FUNC_perf_event_read = 22,
	BPF_FUNC_redirect = 23,
	BPF_FUNC_get_route_realm = 24,
	BPF_FUNC_perf_event_output = 25,
	BPF_FUNC_skb_load_bytes = 26,
	BPF_FUNC_get_stackid = 27,
	BPF_FUNC_csum_diff = 28,
	BPF_FUNC_skb_get_tunnel_opt = 29,
	BPF_FUNC_skb_set_tunnel_opt = 30,
	BPF_FUNC_skb_change_proto = 31,
	BPF_FUNC_skb_change_type = 32,
	BPF_FUNC_skb_under_cgroup = 33,
	BPF_FUNC_get_hash_recalc = 34,
	BPF_FUNC_get_current_task = 35,
	BPF_FUNC_probe_write_user = 36,
	BPF_FUNC_current_task_under_cgroup = 37,
	BPF_FUNC_skb_change_tail = 38,
	BPF_FUNC_skb_pull_data = 39,
	BPF_FUNC_csum_update = 40,
	BPF_FUNC_set_hash_invalid = 41,
	BPF_FUNC_get_numa_node_id = 42,
	BPF_FUNC_skb_change_head = 43,
	BPF_FUNC_xdp_adjust_head = 44,
	BPF_FUNC_probe_read_str = 45,
	BPF_FUNC_get_socket_cookie = 46,
	BPF_FUNC_get_socket_uid = 47,
	BPF_FUNC_set_hash = 48,
	BPF_FUNC_setsockopt = 49,
	BPF_FUNC_skb_adjust_room = 50,
	BPF_FUNC_redirect_map = 51,
	BPF_FUNC_sk_redirect_map = 52,
	BPF_FUNC_sock_map_update = 53,
	BPF_FUNC_xdp_adjust_meta = 54,
	BPF_FUNC_perf_event_read_value = 55,
	BPF_FUNC_perf_prog_read_value = 56,
	BPF_FUNC_getsockopt = 57,
	BPF_FUNC_override_return = 58,
	BPF_FUNC_sock_ops_cb_flags_set = 59,
	BPF_FUNC_msg_redirect_map = 60,
	BPF_FUNC_msg_apply_bytes = 61,
	BPF_FUNC_msg_cork_bytes = 62,
	BPF_FUNC_msg_pull_data = 63,
	BPF_FUNC_bind = 64,
	BPF_FUNC_xdp_adjust_tail = 65,
	BPF_FUNC_skb_get_xfrm_state = 66,
	BPF_FUNC_get_stack = 67,
	BPF_FUNC_skb_load_bytes_relative = 68,
	BPF_FUNC_fib_lookup = 69,
	BPF_FUNC_sock_hash_update = 70,
	BPF_FUNC_msg_redirect_hash = 71,
	BPF_FUNC_sk_redirect_hash = 72,
	BPF_FUNC_lwt_push_encap = 73,
	BPF_FUNC_lwt_seg6_store_bytes = 74,
	BPF_FUNC_lwt_seg6_adjust_srh = 75,
	BPF_FUNC_lwt_seg6_action = 76,
	BPF_FUNC_rc_repeat = 77,
	BPF_FUNC_rc_keydown = 78,
	BPF_FUNC_skb_cgroup_id = 79,
	BPF_FUNC_get_current_cgroup_id = 80,
	BPF_FUNC_get_local_storage = 81,
	BPF_FUNC_sk_select_reuseport = 82,
	BPF_FUNC_skb_ancestor_cgroup_id = 83,
	BPF_FUNC_sk_lookup_tcp = 84,
	BPF_FUNC_sk_lookup_udp = 85,
	BPF_FUNC_sk_release = 86,
	BPF_FUNC_map_push_elem = 87,
	BPF_FUNC_map_pop_elem = 88,
	BPF_FUNC_map_peek_elem = 89,
	BPF_FUNC_msg_push_data = 90,
	BPF_FUNC_msg_pop_data = 91,
	BPF_FUNC_rc_pointer_rel = 92,
	BPF_FUNC_spin_lock = 93,
	BPF_FUNC_spin_unlock = 94,
	BPF_FUNC_sk_fullsock = 95,
	BPF_FUNC_tcp_sock = 96,
	BPF_FUNC_skb_ecn_set_ce = 97,
	BPF_FUNC_get_listener_sock = 98,
	BPF_FUNC_skc_lookup_tcp = 99,
	BPF_FUNC_tcp_check_syncookie = 100,
	BPF_FUNC_sysctl_get_name = 101,
	BPF_FUNC_sysctl_get_current_value = 102,
	BPF_FUNC_sysctl_get_new_value = 103,
	BPF_FUNC_sysctl_set_new_value = 104,
	BPF_FUNC_strtol = 105,
	BPF_FUNC_strtoul = 106,
	BPF_FUNC_sk_storage_get = 107,
	BPF_FUNC_sk_storage_delete = 108,
	BPF_FUNC_send_signal = 109,
	BPF_FUNC_tcp_gen_syncookie = 110,
	BPF_FUNC_skb_output = 111,
	BPF_FUNC_probe_read_user = 112,
	BPF_FUNC_probe_read_kernel = 113,
	BPF_FUNC_probe_read_user_str = 114,
	BPF_FUNC_probe_read_kernel_str = 115,
	BPF_FUNC_tcp_send_ack = 116,
	BPF_FUNC_send_signal_thread = 117,
	BPF_FUNC_jiffies64 = 118,
	BPF_FUNC_read_branch_records = 119,
	BPF_FUNC_get_ns_current_pid_tgid = 120,
	BPF_FUNC_xdp_output = 121,
	BPF_FUNC_get_netns_cookie = 122,
	BPF_FUNC_get_current_ancestor_cgroup_id = 123,
	BPF_FUNC_sk_assign = 124,
	BPF_FUNC_ktime_get_boot_ns = 125,
	BPF_FUNC_seq_printf = 126,
	BPF_FUNC_seq_write = 127,
	BPF_FUNC_sk_cgroup_id = 128,
	BPF_FUNC_sk_ancestor_cgroup_id = 129,
	BPF_FUNC_ringbuf_output = 130,
	BPF_FUNC_ringbuf_reserve = 131,
	BPF_FUNC_ringbuf_submit = 132,
	BPF_FUNC_ringbuf_discard = 133,
	BPF_FUNC_ringbuf_query = 134,
	BPF_FUNC_csum_level = 135,
	BPF_FUNC_skc_to_tcp6_sock = 136,
	BPF_FUNC_skc_to_tcp_sock = 137,
	BPF_FUNC_skc_to_tcp_timewait_sock = 138,
	BPF_FUNC_skc_to_tcp_request_sock = 139,
	BPF_FUNC_skc_to_udp6_sock = 140,
	BPF_FUNC_get_task_stack = 141,
	BPF_FUNC_load_hdr_opt = 142,
	BPF_FUNC_store_hdr_opt = 143,
	BPF_FUNC_reserve_hdr_opt = 144,
	BPF_FUNC_inode_storage_get = 145,
	BPF_FUNC_inode_storage_delete = 146,
	BPF_FUNC_d_path = 147,
	BPF_FUNC_copy_from_user = 148,
	BPF_FUNC_snprintf_btf = 149,
	BPF_FUNC_seq_printf_btf = 150,
	BPF_FUNC_skb_cgroup_classid = 151,
	BPF_FUNC_redirect_neigh = 152,
	BPF_FUNC_per_cpu_ptr = 153,
	BPF_FUNC_this_cpu_ptr = 154,
	BPF_FUNC_redirect_peer = 155,
	BPF_FUNC_task_storage_get = 156,
	BPF_FUNC_task_storage_delete = 157,
	BPF_FUNC_get_current_task_btf = 158,
	BPF_FUNC_bprm_opts_set = 159,
	BPF_FUNC_ktime_get_coarse_ns = 160,
	BPF_FUNC_ima_inode_hash = 161,
	BPF_FUNC_sock_from_file = 162,
	BPF_FUNC_check_mtu = 163,
	BPF_FUNC_for_each_map_elem = 164,
	BPF_FUNC_snprintf = 165,
	BPF_FUNC_sys_bpf = 166,
	BPF_FUNC_btf_find_by_name_kind = 167,
	BPF_FUNC_sys_close = 168,
	BPF_FUNC_timer_init = 169,
	BPF_FUNC_timer_set_callback = 170,
	BPF_FUNC_timer_start = 171,
	BPF_FUNC_timer_cancel = 172,
	BPF_FUNC_get_func_ip = 173,
	BPF_FUNC_get_attach_cookie = 174,
	BPF_FUNC_task_pt_regs = 175,
	BPF_FUNC_get_branch_snapshot = 176,
	BPF_FUNC_trace_vprintk = 177,
	BPF_FUNC_skc_to_unix_sock = 178,
	BPF_FUNC_kallsyms_lookup_name = 179,
	BPF_FUNC_find_vma = 180,
	BPF_FUNC_loop = 181,
	BPF_FUNC_strncmp = 182,
	BPF_FUNC_get_func_arg = 183,
	BPF_FUNC_get_func_ret = 184,
	BPF_FUNC_get_func_arg_cnt = 185,
	BPF_FUNC_get_retval = 186,
	BPF_FUNC_set_retval = 187,
	BPF_FUNC_xdp_get_buff_len = 188,
	BPF_FUNC_xdp_load_bytes = 189,
	BPF_FUNC_xdp_store_bytes = 190,
	BPF_FUNC_copy_from_user_task = 191,
	BPF_FUNC_skb_set_tstamp = 192,
	BPF_FUNC_ima_file_hash = 193,
	BPF_FUNC_kptr_xchg = 194,
	BPF_FUNC_map_lookup_percpu_elem = 195,
	BPF_FUNC_skc_to_mptcp_sock = 196,
	BPF_FUNC_dynptr_from_mem = 197,
	BPF_FUNC_ringbuf_reserve_dynptr = 198,
	BPF_FUNC_ringbuf_submit_dynptr = 199,
	BPF_FUNC_ringbuf_discard_dynptr = 200,
	BPF_FUNC_dynptr_read = 201,
	BPF_FUNC_dynptr_write = 202,
	BPF_FUNC_dynptr_data = 203,
	BPF_FUNC_tcp_raw_gen_syncookie_ipv4 = 204,
	BPF_FUNC_tcp_raw_gen_syncookie_ipv6 = 205,
	BPF_FUNC_tcp_raw_check_syncookie_ipv4 = 206,
	BPF_FUNC_tcp_raw_check_syncookie_ipv6 = 207,
	BPF_FUNC_ktime_get_tai_ns = 208,
	BPF_FUNC_user_ringbuf_drain = 209,
	BPF_FUNC_cgrp_storage_get = 210,
	BPF_FUNC_cgrp_storage_delete = 211,
	__BPF_FUNC_MAX_ID = 212,
};

int bpftime_helper_group::register_helper(const bpftime_helper_info &info)
{
	if (info.index > 999) {
		spdlog::error("Helper id should be 0-999, found {}",
			      info.index);
		return -1;
	}
	if (helper_map.find(info.index) == helper_map.end()) {
		helper_map[info.index] = info;
	} else {
		// found the same helper id
		if (helper_map[info.index].fn != info.fn) {
			spdlog::error("Helper id already exists for {}",
				      info.name);
			return -1;
		}
		// else, ignore the same helper
	}
	return 0;
}

int bpftime_helper_group::append(const bpftime_helper_group &another_group)
{
	for (const auto &it : another_group.helper_map) {
		if (helper_map.find(it.first) != helper_map.end()) {
			spdlog::error("Helper id already exists for {}",
				      it.second.name);
			return -1;
		}
	}
	helper_map.insert(another_group.helper_map.begin(),
			  another_group.helper_map.end());
	return 0;
}

int bpftime_helper_group::add_helper_group_to_prog(bpftime_prog *prog) const
{
	for (auto it : helper_map) {
		prog->bpftime_prog_register_raw_helper(it.second);
	}
	return 0;
}
std::vector<int32_t> bpftime_helper_group::get_helper_ids() const
{
	std::vector<int32_t> result;
	for (const auto &[k, _] : this->helper_map) {
		result.push_back(k);
	}
	return result;
}
const bpftime_helper_group shm_maps_group = { {
	{ BPF_FUNC_map_lookup_elem,
	  bpftime_helper_info{
		  .index = BPF_FUNC_map_lookup_elem,
		  .name = "bpf_map_lookup_elem",
		  .fn = (void *)bpftime_map_lookup_elem_helper,
	  } },
	{ BPF_FUNC_map_update_elem,
	  bpftime_helper_info{
		  .index = BPF_FUNC_map_update_elem,
		  .name = "bpf_map_update_elem",
		  .fn = (void *)bpftime_map_update_elem_helper,
	  } },
	{ BPF_FUNC_map_delete_elem,
	  bpftime_helper_info{
		  .index = BPF_FUNC_map_delete_elem,
		  .name = "bpf_map_delete_elem",
		  .fn = (void *)bpftime_map_delete_elem_helper,
	  } },
} };

const bpftime_helper_group ffi_group = { {
	{ FFI_HELPER_ID_FIND_ID,
	  bpftime_helper_info{
		  .index = FFI_HELPER_ID_FIND_ID,
		  .name = "__ebpf_call_find_ffi_id",
		  .fn = (void *)__ebpf_call_find_ffi_id,
	  } },
	{ FFI_HELPER_ID_DISPATCHER,
	  bpftime_helper_info{
		  .index = FFI_HELPER_ID_DISPATCHER,
		  .name = "__ebpf_call_ffi_dispatcher",
		  .fn = (void *)__ebpf_call_ffi_dispatcher,
	  } },
} };

const bpftime_helper_group kernel_helper_group = {
	{ { BPF_FUNC_probe_read,
	    bpftime_helper_info{
		    .index = BPF_FUNC_probe_read,
		    .name = "bpf_probe_read",
		    .fn = (void *)bpftime_probe_read,
	    } },
	  { BPF_FUNC_probe_read_kernel,
	    bpftime_helper_info{
		    .index = BPF_FUNC_probe_read_kernel,
		    .name = "bpf_probe_read_kernel",
		    .fn = (void *)bpftime_probe_read,
	    } },
	  { BPF_FUNC_probe_read_user,
	    bpftime_helper_info{
		    .index = BPF_FUNC_probe_read_user,
		    .name = "bpf_probe_read_user",
		    .fn = (void *)bpftime_probe_read,
	    } },
	  { BPF_FUNC_ktime_get_ns,
	    bpftime_helper_info{
		    .index = BPF_FUNC_ktime_get_ns,
		    .name = "bpf_ktime_get_ns",
		    .fn = (void *)bpftime_ktime_get_ns,
	    } },
	  { BPF_FUNC_trace_printk,
	    bpftime_helper_info{
		    .index = BPF_FUNC_trace_printk,
		    .name = "bpf_trace_printk",
		    .fn = (void *)bpftime_trace_printk,
	    } },
	  { BPF_FUNC_get_current_pid_tgid,
	    bpftime_helper_info{
		    .index = BPF_FUNC_get_current_pid_tgid,
		    .name = "bpf_get_current_pid_tgid",
		    .fn = (void *)bpftime_get_current_pid_tgid,
	    } },
	  { BPF_FUNC_get_current_uid_gid,
	    bpftime_helper_info{ .index = BPF_FUNC_get_current_uid_gid,
				 .name = "bpf_get_current_uid_gid",
				 .fn = (void *)bpf_get_current_uid_gid } },
	  { BPF_FUNC_get_current_comm,
	    bpftime_helper_info{
		    .index = BPF_FUNC_get_current_comm,
		    .name = "bpf_get_current_comm",
		    .fn = (void *)bpftime_get_current_comm,
	    } },
	  { BPF_FUNC_strncmp,
	    bpftime_helper_info{
		    .index = BPF_FUNC_strncmp,
		    .name = "bpf_strncmp",
		    .fn = (void *)bpftime_strncmp,
	    } },
	  { BPF_FUNC_get_func_arg,
	    bpftime_helper_info{
		    .index = BPF_FUNC_get_func_arg,
		    .name = "bpf_get_func_arg",
		    .fn = (void *)bpftime_get_func_arg,
	    } },
	  { BPF_FUNC_get_func_ret,
	    bpftime_helper_info{
		    .index = BPF_FUNC_get_func_ret,
		    .name = "bpf_get_func_ret_id",
		    .fn = (void *)bpftime_get_func_ret,
	    } },
	  { BPF_FUNC_get_retval,
	    bpftime_helper_info{
		    .index = BPF_FUNC_get_retval,
		    .name = "bpf_get_retval",
		    .fn = (void *)bpftime_get_retval,
	    } },
	  { BPF_FUNC_set_retval,
	    bpftime_helper_info{
		    .index = BPF_FUNC_set_retval,
		    .name = "bpf_set_retval",
		    .fn = (void *)bpftime_set_retval,
	    } },
	  { BPF_FUNC_probe_read_user_str,
	    bpftime_helper_info{
		    .index = BPF_FUNC_probe_read_user_str,
		    .name = "bpf_probe_read_str",
		    .fn = (void *)bpf_probe_read_str,
	    } },
	  { BPF_FUNC_probe_read_str,
	    bpftime_helper_info{
		    .index = BPF_FUNC_probe_read_str,
		    .name = "bpf_probe_str",
		    .fn = (void *)bpf_probe_read_str,
	    } },
	  { BPF_FUNC_get_stack,
	    bpftime_helper_info{ .index = BPF_FUNC_get_stack,
				 .name = "bpf_get_stack",
				 .fn = (void *)bpf_get_stack } },
	  { BPF_FUNC_ktime_get_coarse_ns,
	    bpftime_helper_info{ .index = BPF_FUNC_ktime_get_coarse_ns,
				 .name = "bpf_ktime_get_coarse_ns",
				 .fn = (void *)bpf_ktime_get_coarse_ns } },

	  { BPF_FUNC_ringbuf_reserve,
	    bpftime_helper_info{
		    .index = BPF_FUNC_ringbuf_reserve,
		    .name = "bpf_ringbuf_reserve",
		    .fn = (void *)bpf_ringbuf_reserve,
	    } },
	  { BPF_FUNC_ringbuf_submit,
	    bpftime_helper_info{
		    .index = BPF_FUNC_ringbuf_submit,
		    .name = "bpf_ringbuf_submit",
		    .fn = (void *)bpf_ringbuf_submit,
	    } },
	  { BPF_FUNC_ringbuf_discard,
	    bpftime_helper_info{
		    .index = BPF_FUNC_ringbuf_discard,
		    .name = "bpf_ringbuf_discard",
		    .fn = (void *)bpf_ringbuf_discard,
	    } },
	  { BPF_FUNC_perf_event_output,
	    bpftime_helper_info{ .index = BPF_FUNC_perf_event_output,
				 .name = "bpf_perf_event_output",
				 .fn = (void *)bpf_perf_event_output } },
	  { BPF_FUNC_ringbuf_output,
	    bpftime_helper_info{ .index = BPF_FUNC_ringbuf_output,
				 .name = "bpf_ringbuf_output",
				 .fn = (void *)bpf_ringbuf_output } } }
};

// Utility function to get the FFI helper group
const bpftime_helper_group &bpftime_helper_group::get_ffi_helper_group()
{
	return ffi_group;
}

// Utility function to get the kernel utilities helper group
const bpftime_helper_group &
bpftime_helper_group::get_kernel_utils_helper_group()
{
	return kernel_helper_group;
}

const bpftime_helper_group &bpftime_helper_group::get_shm_maps_helper_group()
{
	return shm_maps_group;
}

#ifdef ENABLE_BPFTIME_VERIFIER
std::map<int32_t, verifier::BpftimeHelperProrotype> get_ffi_helper_protos()
{
	using namespace verifier;
	std::map<int32_t, BpftimeHelperProrotype> result;
	result[FFI_HELPER_ID_FIND_ID] = BpftimeHelperProrotype{
		.name = "__ebpf_call_find_ffi_id",
		.return_type = EBPF_RETURN_TYPE_INTEGER,
		.argument_type = { EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM_OR_NULL,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE }
	};

	result[FFI_HELPER_ID_DISPATCHER] = BpftimeHelperProrotype{
		.name = "__ebpf_call_ffi_dispatcher",
		.return_type = EBPF_RETURN_TYPE_INTEGER,
		.argument_type = { EBPF_ARGUMENT_TYPE_ANYTHING,
				   EBPF_ARGUMENT_TYPE_PTR_TO_READABLE_MEM_OR_NULL,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE,
				   EBPF_ARGUMENT_TYPE_DONTCARE }
	};

	return result;
}
#endif

} // namespace bpftime
