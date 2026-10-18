#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int add_func(int a, int b)
{
	return a + b;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	while (1) {
		int a = rand() & 0xff;
		int b = rand() & 0xff;
		int c = add_func(a, b);
		printf("%d + %d = %d\n", a, b, c);
        usleep(1000 * 1000);
        fflush(stdout);
	}
	return 0;
}
