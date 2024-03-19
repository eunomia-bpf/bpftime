#include ".output/uprobe_multi.skel.h"
#include "bpf/libbpf.h"
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

int main()
{
    // libbpf_set_print(libbpf_print_fn);
	int err;
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);
	struct uprobe_multi_bpf *skel = uprobe_multi_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}
	err = uprobe_multi_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = uprobe_multi_bpf__attach(skel);
	if (err < 0) {
		fprintf(stderr, "Unable to attach %d", err);
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		puts("See /sys/kernel/tracing/trace_pipe");
	}
	// bpf_program__attach_uprobe_multi(const struct bpf_program *prog,
	// pid_t pid, const char *binary_path, const char *func_pattern, const
	// struct bpf_uprobe_multi_opts *opts)
cleanup:
	uprobe_multi_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
