/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

typedef int (*target_fn_t)(int);

int main(int argc, char **argv)
{
	if (argc != 3)
		return 64;
	const char *lib_path = argv[1];
	int pipe_fd = atoi(argv[2]);
	if (pipe_fd <= 0)
		return 65;
	alarm(30);
	/* Even with a uprobe registered in SHM before exec, the unmapped DSO must
	 * not be force-loaded into /proc/self/maps at startup. */
	FILE *maps = fopen("/proc/self/maps", "r");
	if (!maps)
		return 66;
	char line[4096];
	while (fgets(line, sizeof(line), maps)) {
		if (strstr(line, lib_path))
			return 67;
	}
	fclose(maps);
	if (getpgid(0) < 0)
		return 68;
	puts("ready");
	fflush(stdout);

	target_fn_t fn = NULL;
	char step = 0;
	while (read(pipe_fd, &step, 1) == 1 && step != 'q') {
		if (step == 'l') {
			errno = E2BIG;
			void *handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
			if (!handle || errno != E2BIG)
				return 69;
			fn = (target_fn_t)dlsym(handle,
						"bpftime_dlopen_target_func");
			pid_t child = fork();
			if (child < 0)
				return 70;
			if (child == 0) {
				alarm(5);
				errno = ENOTTY;
				void *h = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
				_exit(h && errno == ENOTTY ? 0 : 71);
			}
			int status = 0;
			if (waitpid(child, &status, 0) != child ||
			    !WIFEXITED(status) || WEXITSTATUS(status) != 0)
				return 72;
		}
		if (!fn || fn(10) != 11 || getpgid(0) < 0)
			return 73;
		puts("done");
		fflush(stdout);
	}
	return 0;
}
