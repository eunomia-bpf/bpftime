#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/bpf.h>
#include <cstdio>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <linux/perf_event.h>
#include <spdlog/spdlog.h>
#include "handle_bpf_event.hpp"
#include "../bpf_tracer_event.h"

#define PERF_UPROBE_REF_CTR_OFFSET_BITS 32
#define PERF_UPROBE_REF_CTR_OFFSET_SHIFT 32

using namespace bpftime;

/*
 * this function is expected to parse integer in the range of [0, 2^31-1] from
 * given file using scanf format string fmt. If actual parsed value is
 * negative, the result might be indistinguishable from error
 */
static int parse_uint_from_file(const char *file, const char *fmt)
{
	int err, ret;
	FILE *f;

	f = fopen(file, "re");
	if (!f) {
		err = -errno;
		fprintf(stderr, "failed to open '%s\n", file);
		exit(1);
	}
	err = fscanf(f, fmt, &ret);
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		fprintf(stderr, "failed to parse '%s'\n", file);
		fclose(f);
		exit(1);
	}
	fclose(f);
	return ret;
}

int bpftime::determine_kprobe_perf_type(void)
{
	const char *file = "/sys/bus/event_source/devices/kprobe/type";

	return parse_uint_from_file(file, "%d\n");
}

int bpftime::determine_uprobe_perf_type(void)
{
	const char *file = "/sys/bus/event_source/devices/uprobe/type";

	return parse_uint_from_file(file, "%d\n");
}

int bpftime::determine_uprobe_retprobe_bit()
{
	const char *file =
		"/sys/bus/event_source/devices/uprobe/format/retprobe";

	return parse_uint_from_file(file, "config:%d\n");
}

int bpf_event_handler::handle_open_events(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;
	if (!config.show_open) {
		return 0;
	}

	/* print output */
	spdlog::info("OPEN {:<6} {:<16}", e->pid, e->comm);
	return 0;
}

static const char *bpf_cmd_strings[] = {
	"BPF_MAP_CREATE",	 "BPF_MAP_LOOKUP_ELEM",
	"BPF_MAP_UPDATE_ELEM",	 "BPF_MAP_DELETE_ELEM",
	"BPF_MAP_GET_NEXT_KEY",	 "BPF_PROG_LOAD",
	"BPF_OBJ_PIN",		 "BPF_OBJ_GET",
	"BPF_PROG_ATTACH",	 "BPF_PROG_DETACH",
	"BPF_PROG_TEST_RUN",	 "BPF_PROG_GET_NEXT_ID",
	"BPF_MAP_GET_NEXT_ID",	 "BPF_PROG_GET_FD_BY_ID",
	"BPF_MAP_GET_FD_BY_ID",	 "BPF_OBJ_GET_INFO_BY_FD",
	"BPF_PROG_QUERY",	 "BPF_RAW_TRACEPOINT_OPEN",
	"BPF_BTF_LOAD",		 "BPF_BTF_GET_FD_BY_ID",
	"BPF_TASK_FD_QUERY",	 "BPF_MAP_LOOKUP_AND_DELETE_ELEM",
	"BPF_MAP_FREEZE",	 "BPF_BTF_GET_NEXT_ID",
	"BPF_MAP_LOOKUP_BATCH",	 "BPF_MAP_LOOKUP_AND_DELETE_BATCH",
	"BPF_MAP_UPDATE_BATCH",	 "BPF_MAP_DELETE_BATCH",
	"BPF_LINK_CREATE",	 "BPF_LINK_UPDATE",
	"BPF_LINK_GET_FD_BY_ID", "BPF_LINK_GET_NEXT_ID",
	"BPF_ENABLE_STATS",	 "BPF_ITER_CREATE",
	"BPF_LINK_DETACH",	 "BPF_PROG_BIND_MAP",
};

static const char *bpf_prog_type_strings[] = {
	"BPF_PROG_TYPE_UNSPEC",
	"BPF_PROG_TYPE_SOCKET_FILTER",
	"BPF_PROG_TYPE_KPROBE",
	"BPF_PROG_TYPE_SCHED_CLS",
	"BPF_PROG_TYPE_SCHED_ACT",
	"BPF_PROG_TYPE_TRACEPOINT",
	"BPF_PROG_TYPE_XDP",
	"BPF_PROG_TYPE_PERF_EVENT",
	"BPF_PROG_TYPE_CGROUP_SKB",
	"BPF_PROG_TYPE_CGROUP_SOCK",
	"BPF_PROG_TYPE_LWT_IN",
	"BPF_PROG_TYPE_LWT_OUT",
	"BPF_PROG_TYPE_LWT_XMIT",
	"BPF_PROG_TYPE_SOCK_OPS",
	"BPF_PROG_TYPE_SK_SKB",
	"BPF_PROG_TYPE_CGROUP_DEVICE",
	"BPF_PROG_TYPE_SK_MSG",
	"BPF_PROG_TYPE_RAW_TRACEPOINT",
	"BPF_PROG_TYPE_CGROUP_SOCK_ADDR",
	"BPF_PROG_TYPE_LWT_SEG6LOCAL",
	"BPF_PROG_TYPE_LIRC_MODE2",
	"BPF_PROG_TYPE_SK_REUSEPORT",
	"BPF_PROG_TYPE_FLOW_DISSECTOR",
	"BPF_PROG_TYPE_CGROUP_SYSCTL",
	"BPF_PROG_TYPE_RAW_TRACEPOINT_WRITABLE",
	"BPF_PROG_TYPE_CGROUP_SOCKOPT",
	"BPF_PROG_TYPE_TRACING",
	"BPF_PROG_TYPE_STRUCT_OPS",
	"BPF_PROG_TYPE_EXT",
	"BPF_PROG_TYPE_LSM",
	"BPF_PROG_TYPE_SK_LOOKUP",
	"BPF_PROG_TYPE_SYSCALL",
};

static const char *const bpf_map_type_strings[] = {
	"BPF_MAP_TYPE_UNSPEC",
	"BPF_MAP_TYPE_HASH",
	"BPF_MAP_TYPE_ARRAY",
	"BPF_MAP_TYPE_PROG_ARRAY",
	"BPF_MAP_TYPE_PERF_EVENT_ARRAY",
	"BPF_MAP_TYPE_PERCPU_HASH",
	"BPF_MAP_TYPE_PERCPU_ARRAY",
	"BPF_MAP_TYPE_STACK_TRACE",
	"BPF_MAP_TYPE_CGROUP_ARRAY",
	"BPF_MAP_TYPE_LRU_HASH",
	"BPF_MAP_TYPE_LRU_PERCPU_HASH",
	"BPF_MAP_TYPE_LPM_TRIE",
	"BPF_MAP_TYPE_ARRAY_OF_MAPS",
	"BPF_MAP_TYPE_HASH_OF_MAPS",
	"BPF_MAP_TYPE_DEVMAP",
	"BPF_MAP_TYPE_SOCKMAP",
	"BPF_MAP_TYPE_CPUMAP",
	"BPF_MAP_TYPE_XSKMAP",
	"BPF_MAP_TYPE_SOCKHASH",
	"BPF_MAP_TYPE_CGROUP_STORAGE_DEPRECATED",
	"BPF_MAP_TYPE_CGROUP_STORAGE",
	"BPF_MAP_TYPE_REUSEPORT_SOCKARRAY",
	"BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE",
	"BPF_MAP_TYPE_QUEUE",
	"BPF_MAP_TYPE_STACK",
	"BPF_MAP_TYPE_SK_STORAGE",
	"BPF_MAP_TYPE_DEVMAP_HASH",
	"BPF_MAP_TYPE_STRUCT_OPS",
	"BPF_MAP_TYPE_RINGBUF",
	"BPF_MAP_TYPE_INODE_STORAGE",
	"BPF_MAP_TYPE_TASK_STORAGE",
	"BPF_MAP_TYPE_BLOOM_FILTER",
	"BPF_MAP_TYPE_USER_RINGBUF",
	"BPF_MAP_TYPE_CGRP_STORAGE"
};

#define BPF_MAP_TYPE_MAX                                                       \
	(sizeof(bpf_map_type_strings) / sizeof(bpf_map_type_strings[0]))

static const char *get_bpf_map_type_string(enum bpf_map_type type)
{
	if (type >= 0 && type < BPF_MAP_TYPE_MAX) {
		return bpf_map_type_strings[type];
	}
	return "Unknown";
}

int bpf_event_handler::handle_close_event(const struct event *e)
{
	spdlog::info("CLOSE    {:<6} {:<16} fd:{}", e->pid, e->comm,
		     e->close_data.fd);
	if (config.is_driving_bpftime) {
		driver.bpftime_close_server(e->pid, e->close_data.fd);
	}
	return 0;
}

int bpf_event_handler::handle_bpf_event(const struct event *e)
{
	/* prepare fields */
	const char *cmd_str =
		e->bpf_data.bpf_cmd >= (sizeof(bpf_cmd_strings) /
					sizeof(bpf_cmd_strings[0])) ?
			"UNKNOWN COMMAND" :
			bpf_cmd_strings[e->bpf_data.bpf_cmd];

	spdlog::info("BPF      {:<6} {:<16} cmd:{:<16} ret:{}", e->pid, e->comm,
		     cmd_str, e->bpf_data.ret);

	switch (e->bpf_data.bpf_cmd) {
	case BPF_MAP_CREATE:
		/* code */
		spdlog::info(
			"   BPF_MAP_CREATE map_type:{:<16} map_name:{:<16}",
			get_bpf_map_type_string(
				(enum bpf_map_type)e->bpf_data.attr.map_type),
			e->bpf_data.attr.map_name);
		if (config.is_driving_bpftime && e->bpf_data.ret > 0) {
			bpftime::bpf_map_attr attr;
			attr.type = e->bpf_data.attr.map_type;
			attr.key_size = e->bpf_data.attr.key_size;
			attr.value_size = e->bpf_data.attr.value_size;
			attr.max_ents = e->bpf_data.attr.max_entries;
			attr.flags = e->bpf_data.attr.map_flags;
			attr.btf_id = e->bpf_data.attr.btf_fd;
			attr.btf_key_type_id = e->bpf_data.attr.btf_key_type_id;
			attr.btf_value_type_id =
				e->bpf_data.attr.btf_value_type_id;
			attr.btf_vmlinux_value_type_id =
				e->bpf_data.attr.btf_vmlinux_value_type_id;
			attr.map_extra = e->bpf_data.attr.map_extra;
			attr.ifindex = e->bpf_data.attr.map_ifindex;
			driver.bpftime_maps_create_server(
				e->pid, e->bpf_data.ret,
				e->bpf_data.attr.map_name, attr);
		}
		break;
	case BPF_LINK_CREATE:
		/* code */
		spdlog::info("   BPF_LINK_CREATE prog_fd:{} target_fd:{}",
			     e->bpf_data.attr.link_create.prog_fd,
			     e->bpf_data.attr.link_create.target_fd);
		if (config.is_driving_bpftime && e->bpf_data.ret > 0) {
			return driver.bpftime_link_create_server(
				e->pid, e->bpf_data.ret,
				e->bpf_data.attr.link_create.prog_fd,
				e->bpf_data.attr.link_create.target_fd);
		}
		break;
	case BPF_PROG_LOAD:
		/* code */
		spdlog::info(
			"   BPF_PROG_LOAD prog_type:{:<16} prog_name:{:<16}",
			bpf_prog_type_strings[e->bpf_data.attr.prog_type],
			e->bpf_data.attr.prog_name);
		if (config.is_driving_bpftime && e->bpf_data.ret > 0) {
			event load_prog_event =
				bpf_prog_map[e->bpf_data.attr.insns];
			return driver.bpftime_progs_create_server(
				e->pid, e->bpf_data.ret,
				(ebpf_inst *)
					load_prog_event.bpf_loaded_prog.insns,
				load_prog_event.bpf_loaded_prog.insn_cnt,
				e->bpf_data.attr.prog_name,
				e->bpf_data.attr.prog_type);
		}
		break;
	default:
		break;
	}

	return 0;
}

#define PERF_TYPE_MAX_ID 16
static const char *perf_type_id_strings[PERF_TYPE_MAX_ID] = {
	"PERF_TYPE_HARDWARE", "PERF_TYPE_SOFTWARE", "PERF_TYPE_TRACEPOINT",
	"PERF_TYPE_HW_CACHE", "PERF_TYPE_RAW",	    "PERF_TYPE_BREAKPOINT",
};

int bpf_event_handler::handle_perf_event(const struct event *e)
{
	const char *type_id_str = "UNKNOWN TYPE";
	unsigned int perf_type = e->perf_event_data.attr.type;
	if (perf_type >= 0 && perf_type < (sizeof(perf_type_id_strings) /
					   sizeof(perf_type_id_strings[0]))) {
		type_id_str = perf_type_id_strings[perf_type];
	}

	/* print output */
	spdlog::info("PERF     {:<6} {:<16} type:{:<16} ret:{}\n", e->pid,
		     e->comm, type_id_str, e->perf_event_data.ret);
	if (config.is_driving_bpftime && e->perf_event_data.ret > 0) {
		if (perf_type == (unsigned int)uprobe_type) {
			auto attr = &e->perf_event_data.attr;
			// NO legacy bpf types
			bool retprobe = attr->config &
					(1 << determine_uprobe_retprobe_bit());
			spdlog::debug("retprobe {}", retprobe);
			size_t ref_ctr_off = attr->config >>
					     PERF_UPROBE_REF_CTR_OFFSET_SHIFT;
			const char *name = e->perf_event_data.name_or_path;
			uint64_t offset = e->perf_event_data.offset;
			spdlog::debug("Creating uprobe name {} offset {} "
				      "ref_ctr_off {} attr->config={:x}",
				      name, offset, ref_ctr_off, attr->config);
			driver.bpftime_uprobe_create_server(
				e->pid, e->perf_event_data.ret,
				e->perf_event_data.pid, name, offset, retprobe,
				ref_ctr_off);
		}
	}
	return 0;
}

int bpf_event_handler::handle_load_bpf_prog_event(const struct event *e)
{
	const char *prog_type_str =
		e->bpf_loaded_prog.type >= (sizeof(bpf_prog_type_strings) /
					    sizeof(bpf_prog_type_strings[0])) ?
			"UNKNOWN PROG TYPE" :
			bpf_prog_type_strings[e->bpf_loaded_prog.type];

	const char *prog_name = strlen(e->bpf_loaded_prog.prog_name) > 0 ?
					e->bpf_loaded_prog.prog_name :
					"(none)";

	/* print output */
	spdlog::info(
		"BPF_LOAD {:<6} {:<16} name:{:<16} type:{:<16} insn_cnt:{:<6}",
		e->pid, e->comm, prog_name, prog_type_str,
		e->bpf_loaded_prog.insn_cnt);
	// save the program in the map for later lookup in prog load event
	bpf_prog_map[e->bpf_loaded_prog.insns_ptr] = *e;
	return 0;
}

int bpf_event_handler::handle_ioctl(const struct event *e)
{
	int res;
	int fd = e->ioctl_data.fd;
	int req = e->ioctl_data.req;
	int data = e->ioctl_data.data;
	spdlog::info("IOCTL    {:<6} {:<16} fd:{} req:{} data:{}", e->pid,
		     e->comm, fd, req, data);
	if (req == PERF_EVENT_IOC_ENABLE) {
		spdlog::info("Enabling perf event {}", fd);
		if (config.is_driving_bpftime) {
			return driver.bpftime_perf_event_enable_server(e->pid,
								       fd);
		}
	} else if (req == PERF_EVENT_IOC_DISABLE) {
		spdlog::info("Disabling perf event {}", fd);
		if (config.is_driving_bpftime) {
			return driver.bpftime_perf_event_disable_server(e->pid,
									fd);
		}
	} else if (req == PERF_EVENT_IOC_SET_BPF) {
		spdlog::info("Setting bpf for perf event {} and bpf {}", fd,
			     data);
		if (config.is_driving_bpftime) {
			return driver.bpftime_attach_perf_to_bpf_server(
				e->pid, fd, data);
		}
	}
	return 0;
}

int bpf_event_handler::handle_event(const struct event *e)
{
	// ignore events from self
	if (e->pid == current_pid) {
		return 0;
	}
	switch (e->type) {
	case SYS_OPEN:
		return handle_open_events(e);
		break;
	case SYS_PERF_EVENT_OPEN:
		return handle_perf_event(e);
		break;
	case SYS_BPF:
		return handle_bpf_event(e);
		break;
	case BPF_PROG_LOAD_EVENT:
		return handle_load_bpf_prog_event(e);
		break;
	case SYS_CLOSE:
		return handle_close_event(e);
		break;
	case SYS_IOCTL:
		return handle_ioctl(e);
		break;
	}
	return 0;
}

bpf_event_handler::bpf_event_handler(struct daemon_config config,
				     bpftime_driver &driver)
	: config(config), driver(driver)
{
	current_pid = getpid();
	uprobe_type = determine_uprobe_perf_type();
	if (uprobe_type < 0 || uprobe_type >= PERF_TYPE_MAX_ID) {
		spdlog::error("Failed to determine uprobe perf type");
		exit(1);
	}
	perf_type_id_strings[uprobe_type] = "PERF_TYPE_UPROBE";
	kprobe_type = determine_kprobe_perf_type();
	if (kprobe_type < 0 || kprobe_type >= PERF_TYPE_MAX_ID) {
		spdlog::error("Failed to determine kprobe perf type");
		exit(1);
	}

	perf_type_id_strings[kprobe_type] = "PERF_TYPE_KPROBE";
}