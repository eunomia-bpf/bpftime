#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>

#include "request_filter.h"

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct request_counter);
} counter SEC(".maps");

static inline int str_startswith(const char *main, const char *pat)
{
	int i = 0;
	while (*main == *pat && *main != 0 && *pat != 0 && i++ < 128) {
		main++;
		pat++;
	}
	return *pat == 0;
}

SEC("nginx/ngx_http_request_handler")
int request_filter(struct request_filter_argument *arg)
{
	int result = 0;
	result = str_startswith(arg->url_to_check, arg->accept_prefix);
	
	__u32 key = 0;
	struct request_counter *counter_ptr = bpf_map_lookup_elem(&counter, &key);
	if (counter_ptr) {
		if (result) {
			// __sync_fetch_and_add(&counter_ptr->accepted_count, 1);
			counter_ptr->accepted_count++;
		} else {
			// __sync_fetch_and_add(&counter_ptr->rejected_count, 1);
			counter_ptr->rejected_count++;
		}
	}
	
	return result;
}

char LICENSE[] SEC("license") = "GPL";
