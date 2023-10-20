#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <cstdio>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <linux/perf_event.h>
#include "handle_bpf_event.hpp"
#include "bpf-mocker-event.h"

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
		return err;
	}
	err = fscanf(f, fmt, &ret);
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		fprintf(stderr, "failed to parse '%s'\n", file);
		fclose(f);
		return err;
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

int bpf_event_handler::handle_open_events(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;
	int fd, err;
	if (!config.show_open) {
		return 0;
	}

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);
	if (e->open_data.ret >= 0) {
		fd = e->open_data.ret;
		err = 0;
	} else {
		fd = -1;
		err = -e->open_data.ret;
	}

	/* print output */
	printf("OPEN %-6d %-16s %3d %3d ", e->pid, e->comm, fd, err);
	printf("%s\n", e->open_data.fname);
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

int bpf_event_handler::handle_bpf_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	const char *cmd_str =
		e->bpf_data.bpf_cmd >= (sizeof(bpf_cmd_strings) /
					sizeof(bpf_cmd_strings[0])) ?
			"UNKNOWN COMMAND" :
			bpf_cmd_strings[e->bpf_data.bpf_cmd];

	/* print output */
	printf("BPF %-6d %-16s %-16s %d\n", e->pid, e->comm, cmd_str,
	       e->bpf_data.ret);
	return 0;
}

#define PERF_TYPE_MAX_ID 16
static const char *perf_type_id_strings[PERF_TYPE_MAX_ID] = {
	[PERF_TYPE_HARDWARE] = "PERF_TYPE_HARDWARE",
	[PERF_TYPE_SOFTWARE] = "PERF_TYPE_SOFTWARE",
	[PERF_TYPE_TRACEPOINT] = "PERF_TYPE_TRACEPOINT",
	[PERF_TYPE_HW_CACHE] = "PERF_TYPE_HW_CACHE",
	[PERF_TYPE_RAW] = "PERF_TYPE_RAW",
	[PERF_TYPE_BREAKPOINT] = "PERF_TYPE_BREAKPOINT",
};

int bpf_event_handler::handle_perf_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	const char *type_id_str =
		e->perf_event_data.attr.type >=
				(sizeof(perf_type_id_strings) /
				 sizeof(perf_type_id_strings[0])) ?
			"UNKNOWN TYPE" :
			perf_type_id_strings[e->perf_event_data.attr.type];

	/* print output */
	printf("PERF %-6d %-16s %-8s %d\n", e->pid, e->comm, type_id_str,
	       e->perf_event_data.pid);
	return 0;
}

int bpf_event_handler::handle_load_bpf_prog_event(const struct event *e)
{
	struct tm *tm;
	char ts[32];
	time_t t;

	/* prepare fields */
	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	const char *prog_type_str =
		e->bpf_loaded_prog.type >= (sizeof(bpf_prog_type_strings) /
					    sizeof(bpf_prog_type_strings[0])) ?
			"UNKNOWN PROG TYPE" :
			bpf_prog_type_strings[e->bpf_loaded_prog.type];

	/* print output */
	printf("LOAD %-6d %-16s %-16s %-16s %d\n", e->pid, e->comm,
	       prog_type_str, e->bpf_loaded_prog.prog_name,
	       e->bpf_loaded_prog.insn_cnt);
	return 0;
}

int bpf_event_handler::handle_event(const struct event *e)
{
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
	}
	return 0;
}

bpf_event_handler::bpf_event_handler(struct env config) : config(config)
{
	perf_type_id_strings[determine_uprobe_perf_type()] = "PERF_TYPE_UPROBE";
	perf_type_id_strings[determine_kprobe_perf_type()] = "PERF_TYPE_KPROBE";
}