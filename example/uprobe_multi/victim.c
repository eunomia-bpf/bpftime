#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
long uprobe_multi_func_add(long a, long b)
{
	return a + b;
}
long uprobe_multi_func_sub(long a, long b)
{
	return a - b;
}
long uprobe_multi_func_mul(long a, long b)
{
	return a * b;
}

int main()
{
	srand(time(NULL));
	while (true) {
		long a = rand() & 0xff;
		long b = rand() & 0xff;

		long r1 = uprobe_multi_func_add(a, b);
		long r2 = uprobe_multi_func_sub(a, b);
		long r3 = uprobe_multi_func_mul(a, b);

		printf("%ld+%ld=%ld, %ld-%ld=%ld, %ld*%ld=%ld\n", a, b, r1, a,
		       b, r2, a, b, r3);
		usleep(1000 * 1000);
	}
	return 0;
}
