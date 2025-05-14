#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <signal.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <linux/bpf.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// 如果还没有生成 skeleton 文件，先使用这些定义
#ifndef _CUDA_SCHEDULER_SKEL_H
struct cuda_scheduler_bpf {
	struct bpf_object *obj;
	struct {
		struct bpf_map *call_count;
		struct bpf_map *run_pid_map;
	} maps;
};
#endif

static volatile bool exiting = false;

void handle_sig(int sig)
{
	exiting = true;
}

int main(int argc, char **argv)
{
	struct cuda_scheduler_bpf *skel;
	int err;

	signal(SIGINT, handle_sig);

	// 打开 BPF 对象
	skel = calloc(1, sizeof(*skel));
	if (!skel) {
		fprintf(stderr, "Failed to allocate BPF skeleton\n");
		return 1;
	}

	// 获取 map 文件描述符
	int map_fd = bpf_obj_get("/sys/fs/bpf/call_count");
	if (map_fd < 0) {
		fprintf(stderr, "Failed to get call_count map: %s\n",
			strerror(errno));
		return 1;
	}

	int run_fd = bpf_obj_get("/sys/fs/bpf/run_pid_map");
	if (run_fd < 0) {
		fprintf(stderr, "Failed to get run_pid_map: %s\n",
			strerror(errno));
		return 1;
	}

	printf("Starting C-based fair-share scheduler (Jain's F)\n");
	while (!exiting) {
		// 读取所有 PID 及其调用次数
		__u32 key = 0, next_key;
		__u32 count;
		// first, collect counts
		struct {
			__u32 pid;
			__u64 cnt;
		} entries[1024];
		int n = 0;

		// 遍历 map
		while (bpf_map_get_next_key(map_fd, &key, &next_key) == 0) {
			if (bpf_map_lookup_elem(map_fd, &next_key, &count) ==
			    0) {
				entries[n].pid = next_key;
				entries[n].cnt = count;
				n++;
			}
			key = next_key;
		}

		if (n > 0) {
			__u64 sum = 0, sum2 = 0;
			for (int i = 0; i < n; i++) {
				sum += entries[i].cnt;
				sum2 += entries[i].cnt * entries[i].cnt;
			}
			double bestF = -1;
			__u32 bestPid = 0;
			for (int i = 0; i < n; i++) {
				__u64 c = entries[i].cnt;
				double new_sum = sum + 1;
				double new_sum2 = sum2 - (double)c * c +
						  (double)(c + 1) * (c + 1);
				double F = new_sum * new_sum / (n * new_sum2);
				if (F > bestF) {
					bestF = F;
					bestPid = entries[i].pid;
				}
			}
			// 更新 run_pid_map
			__u32 idx = 0;
			bpf_map_update_elem(run_fd, &idx, &bestPid, BPF_ANY);
		}
		usleep(10000); // 10 ms
	}

	close(map_fd);
	close(run_fd);
	free(skel);
	return 0;
}
