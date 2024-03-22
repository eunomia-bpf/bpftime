#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>

#include "request_filter.h"

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1024 * 256);
} events SEC(".maps");

static inline int str_startswith(const char *main, const char *pat)
{
	int i = 0;
	while (*main == *pat && *main != 0 && *pat != 0 && i++ < 128) {
		main++;
		pat++;
	}
	return *pat == 0;
}

int request_filter(struct request_filter_argument *arg)
{
	int result = 0;
	struct request_filter_event *eventp;
	result = str_startswith(arg->url_to_check, arg->accept_prefix);
	eventp = bpf_ringbuf_reserve(&events, sizeof(*eventp), 0);
	bpf_printk("Allocted buffer %lx", (uintptr_t)eventp);
	if (eventp) {
		for (int i = 0; i < 128; i += 1)
			eventp->url[i] = arg->url_to_check[i];
		eventp->accepted = result;
		bpf_ringbuf_submit(eventp, 0);
	}
	return result;
}

char LICENSE[] SEC("license") = "GPL";
