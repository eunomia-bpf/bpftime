// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
	for (;;) {
		void *ptr = malloc(1024);
		if (!ptr) {
			return 1;
		}
		printf("malloc/free loop\n");
		fflush(stdout);
		usleep(100 * 1000);
		free(ptr);
	}
}
