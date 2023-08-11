#include "bpftime.h"
#include "../agent/agent.h"
#include <stdio.h>

const char *bpf_object_path;

int hello(char* a, int b, uint64_t c)
{
	return a[b] + c;
}

int main()
{
	int res = 0;
	struct bpf_probe_ctx *ctx = bpftime_probe_create_ctx();
	struct agent_context *agent_ctx = create_agent_context();
	struct patch_config *config;
	// agent_install_patch(agent_ctx, config);
	struct bpf_probe_ctx *probe_ctx = bpftime_probe_create_ctx();
	if (probe_ctx == NULL) {
		printf("error: probe_ctx is NULL\n");
		return res;
	}
	struct bpftime_object *obj = bpftime_object_open(bpf_object_path);
	if (!obj) {
		printf("error: %d in line: %d\n", res, __LINE__);
		return res;
	}
	// get the first program
	struct bpftime_prog *prog = bpftime_object__next_program(obj, NULL);
	if (!prog) {
		printf("error: %d in bpftime_object__next_program\n", res);
		return res;
	}
	res = bpftime_prog_load(prog, false);
	if (res != 0) {
		printf("error: %d in bpftime_prog_load", res);
		return res;
	}
	res = bpftime_probe__attach_replace(probe_ctx, prog, (void *)hello);
	if (res != 0) {
		printf("error: %d in bpftime_probe__attach_replace\n", res);
		return res;
	}
	return 0;
}