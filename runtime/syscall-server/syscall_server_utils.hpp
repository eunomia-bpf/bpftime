#ifndef _SYSCALL_SERVER_UTILS_HPP
#define _SYSCALL_SERVER_UTILS_HPP

int determine_uprobe_perf_type();
int determine_uprobe_retprobe_bit();
void start_up();

#define PERF_UPROBE_REF_CTR_OFFSET_BITS 32
#define PERF_UPROBE_REF_CTR_OFFSET_SHIFT 32

#endif
