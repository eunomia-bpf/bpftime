/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int target_func()
{
	printf("target_func is running and write something.\n");
	return 0;
}

int main(int argc, char *argv[])
{
	while (1) {
		sleep(1);
		int res = target_func();
		if (res != 0) {
			printf("got error %d\n", res);
		}
	}
	return 0;
}
