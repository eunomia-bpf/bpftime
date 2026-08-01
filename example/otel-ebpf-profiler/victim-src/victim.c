// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
	uint64_t iteration = 0;

	for (;;) {
		volatile unsigned char *allocation = malloc(1024);
		if (allocation == NULL)
			return 1;
		allocation[0] = (unsigned char)iteration++;
		free((void *)allocation);
	}
}
