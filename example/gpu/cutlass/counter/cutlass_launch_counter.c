#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include "./.output/cutlass_launch_counter.skel.h"

static volatile bool exiting = false;

static void handle_signal(int)
{
	exiting = true;
}

static int print_stat(struct cutlass_launch_counter_bpf *obj)
{
	uint32_t key = 0;
	uint64_t value = 0;
	int fd = bpf_map__fd(obj->maps.cutlass_call_count);
	if (bpf_map_lookup_elem(fd, &key, &value) != 0)
		return -1;
	time_t t = time(NULL);
	struct tm *tm = localtime(&t);
	char ts[16];
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);
	printf("%s CUTLASS launches: %lu\n", ts, value);
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;
	struct cutlass_launch_counter_bpf *skel = NULL;
	int err = 0;

	libbpf_set_print(NULL);
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	skel = cutlass_launch_counter_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open skeleton\n");
		return 1;
	}
	err = cutlass_launch_counter_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load skeleton: %d\n", err);
		goto cleanup;
	}
	err = cutlass_launch_counter_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach skeleton: %d\n", err);
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		print_stat(skel);
	}
cleanup:
	cutlass_launch_counter_bpf__destroy(skel);
	return err;
}

