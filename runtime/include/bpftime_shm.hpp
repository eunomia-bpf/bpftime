#ifndef BPFTIME_SHM_CPP_H
#define BPFTIME_SHM_CPP_H

#include "bpftime_config.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <cstddef>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <ebpf-vm.h>
#include <sys/epoll.h>

namespace bpftime
{

struct bpf_map_attr {
	int type = 0;
	uint32_t key_size = 0;
	uint32_t value_size = 0;
	uint32_t max_ents = 0;
	uint64_t flags = 0;
	uint32_t ifindex = 0;
	uint32_t btf_vmlinux_value_type_id = 0;
	uint32_t btf_id = 0;
	uint32_t btf_key_type_id = 0;
	uint32_t btf_value_type_id = 0;
	uint64_t map_extra = 0;

	// additional fields for bpftime only
	uint32_t kernel_bpf_map_id = 0;
};

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
};

#define KERNEL_USER_MAP_OFFSET 1000

enum class bpf_map_type {
	BPF_MAP_TYPE_UNSPEC,
	BPF_MAP_TYPE_HASH,
	BPF_MAP_TYPE_ARRAY,
	BPF_MAP_TYPE_PROG_ARRAY,
	BPF_MAP_TYPE_PERF_EVENT_ARRAY,
	BPF_MAP_TYPE_PERCPU_HASH,
	BPF_MAP_TYPE_PERCPU_ARRAY,
	BPF_MAP_TYPE_STACK_TRACE,
	BPF_MAP_TYPE_CGROUP_ARRAY,
	BPF_MAP_TYPE_LRU_HASH,
	BPF_MAP_TYPE_LRU_PERCPU_HASH,
	BPF_MAP_TYPE_LPM_TRIE,
	BPF_MAP_TYPE_ARRAY_OF_MAPS,
	BPF_MAP_TYPE_HASH_OF_MAPS,
	BPF_MAP_TYPE_DEVMAP,
	BPF_MAP_TYPE_SOCKMAP,
	BPF_MAP_TYPE_CPUMAP,
	BPF_MAP_TYPE_XSKMAP,
	BPF_MAP_TYPE_SOCKHASH,
	BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED,
	/* BPF_MAP_TYPE_CGROUP_STORAGE is available to bpf programs
	 * attaching to a cgroup. The newer BPF_MAP_TYPE_CGRP_STORAGE is
	 * available to both cgroup-attached and other progs and
	 * supports all functionality provided by
	 * BPF_MAP_TYPE_CGROUP_STORAGE. So mark
	 * BPF_MAP_TYPE_CGROUP_STORAGE deprecated.
	 */
	BPF_MAP_TYPE_CGROUP_STORAGE = BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED,
	BPF_MAP_TYPE_REUSEPORT_SOCKARRAY,
	BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE,
	BPF_MAP_TYPE_QUEUE,
	BPF_MAP_TYPE_STACK,
	BPF_MAP_TYPE_SK_STORAGE,
	BPF_MAP_TYPE_DEVMAP_HASH,
	BPF_MAP_TYPE_STRUCT_OPS,
	BPF_MAP_TYPE_RINGBUF,
	BPF_MAP_TYPE_INODE_STORAGE,
	BPF_MAP_TYPE_TASK_STORAGE,
	BPF_MAP_TYPE_BLOOM_FILTER,
	BPF_MAP_TYPE_USER_RINGBUF,
	BPF_MAP_TYPE_CGRP_STORAGE,

	BPF_MAP_TYPE_KERNEL_USER_HASH =
		KERNEL_USER_MAP_OFFSET + BPF_MAP_TYPE_HASH,
	BPF_MAP_TYPE_KERNEL_USER_ARRAY =
		KERNEL_USER_MAP_OFFSET + BPF_MAP_TYPE_ARRAY,
	BPF_MAP_TYPE_KERNEL_USER_PERCPU_ARRAY =
		KERNEL_USER_MAP_OFFSET + BPF_MAP_TYPE_PERCPU_ARRAY,
	BPF_MAP_TYPE_KERNEL_USER_PERF_EVENT_ARRAY =
		KERNEL_USER_MAP_OFFSET + BPF_MAP_TYPE_PERF_EVENT_ARRAY,

};

enum class shm_open_type {
	SHM_REMOVE_AND_CREATE,
	SHM_OPEN_ONLY,
	SHM_NO_CREATE,
	SHM_CREATE_OR_OPEN,
};

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

extern const shm_open_type global_shm_open_type;

const bpftime::agent_config &bpftime_get_agent_config();
void bpftime_set_agent_config(bpftime::agent_config &cfg);
} // namespace bpftime

extern "C" {
// initialize the global shared memory for store bpf progs and maps
void bpftime_initialize_global_shm(bpftime::shm_open_type type);
// destroy the global shared memory data structure
//
// Note: this will NO remove the global shared memory from system
// 	 use bpftime_remove_global_shm() to remove the global shared memory
void bpftime_destroy_global_shm();
// remove the global shared memory from system
void bpftime_remove_global_shm();

// import the global shared memory from json file
int bpftime_import_global_shm_from_json(const char *filename);
// export the global shared memory to json file
int bpftime_export_global_shm_to_json(const char *filename);
// import a hander to global shared memory from json string
int bpftime_import_shm_handler_from_json(int fd, const char *json_string);

// create a bpf link in the global shared memory
//
// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then the
// function will allocate a new perf event fd.
int bpftime_link_create(int fd, int prog_fd, int target_fd);

// create a bpf prog in the global shared memory
//
// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then the
// function will allocate a new perf event fd.
int bpftime_progs_create(int fd, const ebpf_inst *insn, size_t insn_cnt,
			 const char *prog_name, int prog_type);

// create a bpf map in the global shared memory
//
// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then the
// function will allocate a new perf event fd.
int bpftime_maps_create(int fd, const char *name, bpftime::bpf_map_attr attr);

// get the bpf map info from the global shared memory
int bpftime_map_get_info(int fd, bpftime::bpf_map_attr *out_attr,
			 const char **out_name, bpftime::bpf_map_type *type);

// get the map value size from the global shared memory by fd
uint32_t bpftime_map_value_size_from_syscall(int fd);

// used by bpf_helper to get the next key
int bpftime_helper_map_get_next_key(int fd, const void *key, void *next_key);
// used by bpf_helper to lookup the elem
const void *bpftime_helper_map_lookup_elem(int fd, const void *key);
// used by bpf_helper to update the elem
long bpftime_helper_map_update_elem(int fd, const void *key, const void *value,
				    uint64_t flags);
// used by bpf_helper to delete the elem
long bpftime_helper_map_delete_elem(int fd, const void *key);

// use from bpf syscall to get the next key
int bpftime_map_get_next_key(int fd, const void *key, void *next_key);
// use from bpf syscall to lookup the elem
const void *bpftime_map_lookup_elem(int fd, const void *key);
// use from bpf syscall to update the elem
long bpftime_map_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags);
// use from bpf syscall to delete the elem
long bpftime_map_delete_elem(int fd, const void *key);

// create uprobe in the global shared memory
//
// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then the
// function will allocate a new perf event fd.
int bpftime_uprobe_create(int fd, int pid, const char *name, uint64_t offset,
			  bool retprobe, size_t ref_ctr_off);
// create tracepoint in the global shared memory
//
// @param[fd]: fd is the fd allocated by the kernel. if fd is -1, then the
// function will allocate a new perf event fd.
int bpftime_tracepoint_create(int fd, int pid, int32_t tp_id);

// enable the perf event
int bpftime_perf_event_enable(int fd);

// disable the perf event
int bpftime_perf_event_disable(int fd);

int bpftime_find_minimal_unused_fd();

int bpftime_attach_perf_to_bpf(int perf_fd, int bpf_fd);
int bpftime_add_ringbuf_fd_to_epoll(int ringbuf_fd, int epoll_fd,
				    epoll_data_t extra_data);
int bpftime_add_software_perf_event_fd_to_epoll(int swpe_fd, int epoll_fd,
						epoll_data_t extra_data);

int bpftime_epoll_create();
void *bpftime_get_ringbuf_consumer_page(int ringbuf_fd);
void *bpftime_get_ringbuf_producer_page(int ringbuf_fd);

int bpftime_is_ringbuf_map(int fd);
int bpftime_is_array_map(int fd);
int bpftime_is_epoll_handler(int fd);

int bpftime_is_prog_fd(int fd);
int bpftime_is_map_fd(int fd);

void *bpftime_get_array_map_raw_data(int fd);

void bpftime_close(int fd);

void *bpftime_ringbuf_reserve(int fd, uint64_t size);
void bpftime_ringbuf_submit(int fd, void *data, int discard);
int bpftime_epoll_wait(int fd, struct epoll_event *out_evts, int max_evt,
		       int timeout);

int bpftime_add_software_perf_event(int cpu, int32_t sample_type,
				    int64_t config);
int bpftime_is_software_perf_event(int fd);
void *bpftime_get_software_perf_event_raw_buffer(int fd, size_t expected_size);
int bpftime_perf_event_output(int fd, const void *buf, size_t sz);
int bpftime_shared_perf_event_output(int map_fd, const void *buf, size_t sz);
}

#endif // BPFTIME_SHM_CPP_H
