// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <linux/bpf.h>
#include "uprobe_stack.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

// 事件数据结构 (与eBPF程序中的结构保持一致)
struct event_data {
	uint64_t timestamp; // 时间戳
	uint32_t pid; // 进程ID
	uint32_t tid; // 线程ID
	uint32_t counter; // 函数调用计数器
	uint32_t function_id; // 函数标识符 (1=target_function,
			      // 2=secondary_function)
	int32_t input_value; // target_function的输入值
	char comm[16]; // 进程名称
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t stop;

static void sig_int(int signo)
{
	stop = 1;
}

// 使用syscall实现map_pop_elem (bpftime会拦截这个syscall)
static long map_pop_elem_syscall(int fd, void *value)
{
	union bpf_attr attr = {};
	attr.map_fd = fd;
	attr.value = (uint64_t)(unsigned long)value;

	return syscall(__NR_bpf, BPF_MAP_LOOKUP_AND_DELETE_ELEM, &attr,
		       sizeof(attr));
}

// 使用syscall实现map_peek_elem (bpftime会拦截这个syscall)
static long map_peek_elem_syscall(int fd, void *value)
{
	union bpf_attr attr = {};
	attr.map_fd = fd;
	attr.value = (uint64_t)(unsigned long)value;

	return syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr));
}

// 从栈中处理事件 (注意：栈是LIFO，后进先出)
static int process_stack_events(int stack_fd)
{
	struct event_data event;
	int events_processed = 0;

	printf("=== Starting to pop events from stack (LIFO order) ===\n");

	// 持续从栈中弹出事件，直到栈为空
	while (1) {
		// 使用syscall方式，bpftime会拦截并处理
		int ret = map_pop_elem_syscall(stack_fd, &event);
		if (ret != 0) {
			if (ret == -ENOENT || errno == ENOENT) {
				// 栈为空，正常退出
				break;
			} else {
				warn("Failed to pop from stack: %d (errno: %d)\n",
				     ret, errno);
				return -1;
			}
		}

		events_processed++;

		// 格式化时间戳
		time_t timestamp_sec = event.timestamp / 1000000000ULL;
		uint64_t timestamp_nsec = event.timestamp % 1000000000ULL;
		struct tm *tm_info = localtime(&timestamp_sec);
		char time_str[64];
		strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_info);

		// 打印事件信息 (注意这是LIFO顺序，最新的事件先出来)
		printf("[%s.%03lu] [stack pop #%d] ", time_str,
		       timestamp_nsec / 1000000, events_processed);

		if (event.function_id == 1) {
			printf("target_function() - PID:%u TID:%u input:%d counter:%u process:%s\n",
			       event.pid, event.tid, event.input_value,
			       event.counter, event.comm);
		} else if (event.function_id == 2) {
			printf("secondary_function() - PID:%u TID:%u counter:%u process:%s\n",
			       event.pid, event.tid, event.counter, event.comm);
		} else {
			printf("unknown function (ID:%u) - PID:%u TID:%u counter:%u process:%s\n",
			       event.function_id, event.pid, event.tid,
			       event.counter, event.comm);
		}
	}

	if (events_processed > 0) {
		printf("=== Stack pop completed, processed %d events ===\n",
		       events_processed);
	}

	return events_processed;
}

// 显示栈统计信息
static void show_stack_stats(int stack_fd)
{
	struct event_data temp_event;

	// 尝试peek栈顶元素 (使用lookup，不删除元素)
	int peek_ret = map_peek_elem_syscall(stack_fd, &temp_event);

	if (peek_ret == 0) {
		printf("Stack status: non-empty (top event: function_id=%u, counter=%u)\n",
		       temp_event.function_id, temp_event.counter);
	} else if (peek_ret == -ENOENT || errno == ENOENT) {
		printf("Stack status: empty\n");
	} else {
		printf("Stack status: query failed (error: %d, errno: %d)\n",
		       peek_ret, errno);
	}
}

int main(int argc, char **argv)
{
	struct uprobe_stack_bpf *skel;
	int err;
	int stack_fd;

	/* 设置libbpf错误和调试信息回调 */
	libbpf_set_print(libbpf_print_fn);

	/* 打开BPF应用程序 */
	skel = uprobe_stack_bpf__open();
	if (!skel) {
		warn("Failed to open BPF skeleton\n");
		return 1;
	}

	/* 加载并验证BPF程序 */
	err = uprobe_stack_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* 附加uprobe */
	err = uprobe_stack_bpf__attach(skel);
	if (err) {
		warn("Failed to attach BPF program: %d\n", err);
		goto cleanup;
	}

	/* 获取栈map的文件描述符 */
	stack_fd = bpf_map__fd(skel->maps.events_stack);
	if (stack_fd < 0) {
		warn("Failed to get stack map fd\n");
		err = -1;
		goto cleanup;
	}

	printf("Stack Map FD: %d\n", stack_fd);
	printf("eBPF program successfully attached to uprobe\n");
	printf("Starting stack event monitoring...\n");
	printf("Note: Stack is LIFO (Last In First Out), newest events pop first\n");
	printf("Press Ctrl+C to stop\n\n");

	/* 设置信号处理器 */
	if (signal(SIGINT, sig_int) == SIG_ERR) {
		warn("Cannot set signal handler\n");
		err = 1;
		goto cleanup;
	}

	/* 主事件循环 */
	while (!stop) {
		// 显示栈状态
		show_stack_stats(stack_fd);

		// 处理栈中的事件
		int events_count = process_stack_events(stack_fd);
		if (events_count < 0) {
			warn("Error processing stack events\n");
			break;
		}

		if (events_count > 0) {
			printf("Processed %d events this round\n\n",
			       events_count);
		}

		// 等待一段时间再次检查
		sleep(1);
	}

cleanup:
	uprobe_stack_bpf__destroy(skel);
	printf("\nProgram exiting\n");
	return -err;
}