#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
int uprobe_multi_func_add(int a, int b)
{
	return a + b;
}
int uprobe_multi_func_sub(int a, int b)
{
	return a - b;
}
int uprobe_multi_func_mul(int a, int b)
{
	return a * b;
}

int main()
{
	srand(time(NULL));
	while (true) {
		int a = rand() & 0xff;
		int b = rand() & 0xff;

		int r1 = uprobe_multi_func_add(a, b);
		int r2 = uprobe_multi_func_sub(a, b);
		int r3 = uprobe_multi_func_mul(a, b);

		printf("%d+%d=%d, %d-%d=%d, %d*%d=%d\n", a, b, r1, a, b, r2, a,
		       b, r3);
		usleep(1000 * 1000);
	}
	return 0;
}
