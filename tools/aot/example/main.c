#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>

int bpf_main(void* ctx, uint64_t size);

// bpf_printk
uint64_t _bpf_helper_ext_0006(uint64_t fmt, uint64_t fmt_size, ...)
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

// bpf_get_current_pid_tgid
uint64_t _bpf_helper_ext_0014(void)
{
	static int tgid = -1;
	static int tid = -1;
	if (tid == -1)
		tid = gettid();
	if (tgid == -1)
		tgid = getpid();
	return ((uint64_t)tgid << 32) | tid;
}

// here we use an var to mock the map.
uint64_t counter_map = 0;

// bpf_map_lookup_elem
void * _bpf_helper_ext_0001(void *map, const void *key)
{
    printf("bpf_map_lookup_elem\n");
    return &counter_map;
}

// bpf_map_update_elem
long _bpf_helper_ext_0002(void *map, const void *key, const void *value, uint64_t flags)
{
    printf("bpf_map_update_elem\n");
	if (value == NULL) {
		printf("value is NULL\n");
		return -1;
	}
	uint64_t* value_ptr = (uint64_t*)value_ptr;
	counter_map = *value_ptr;
	printf("counter_map: %lu\n", counter_map);
    return 0;
}

uint64_t __lddw_helper_map_by_fd(uint32_t id) {
	printf("map_by_fd\n");
	return 0;
}

int main() {
    printf("Hello, World!\n");
    bpf_main(NULL, 0);
    return 0;
}
