// SPDX-License-Identifier: MIT
#include <stdlib.h>

int main(void)
{
	for (;;) {
		volatile unsigned char *allocation = malloc(1024);
		if (allocation == NULL)
			return 1;
		allocation[0] = 0;
		free((void *)allocation);
	}
}
