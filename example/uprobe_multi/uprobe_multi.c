#include ".output/uprobe_multi.skel.h"
#include "bpf/libbpf.h"
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include "uprobe_multi.h"
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

static int handle_event_rb(void *ctx, void *data, size_t data_sz)
{
	const struct uprobe_multi_event *e = data;
	if (!e->is_ret) {
		printf("Uprobe triggered: %ld , %ld\n", e->uprobe.arg1,
		       e->uprobe.arg2);
	} else {
		printf("Uretprobe triggered: %ld \n", e->uretprobe.ret_val);
	}
	fflush(stdout);
	return 0;
}

int main()
{
	libbpf_set_print(libbpf_print_fn);
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
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.rb),
						  handle_event_rb, NULL, NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	while (!exiting) {
		err = ring_buffer__poll(rb, 100);
		if (err < 0 && err != -EINTR) {
			fprintf(stderr, "error polling perf buffer: %s\n",
				strerror(-err));
			goto cleanup;
		}
		err = 0;
	}
cleanup:
	uprobe_multi_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
