#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>

int my_open(const char *pathname, int flags)
{
	return open(pathname, flags);
}

int my_close(int fd)
{
	return close(fd);
}

void test_open(void)
{
	int fd;

	fd = my_open("/etc/hosts", O_RDONLY);
	if (fd != -1)
		my_close(fd);

	fd = my_open("/etc/passwd", O_RDONLY);
	if (fd != -1)
		my_close(fd);
}

int test_calc_fib(int pid)
{
	int t1 = 0, t2 = 1, nextTerm = 0;

	for (int i = 1; i <= pid; ++i) {
		// Prints the first two terms.
		if (i == 1) {
			continue;
		}
		if (i == 2) {
			continue;
		}
		nextTerm = t1 + t2;
		t1 = t2;
		t2 = nextTerm;
	}
	return t2;
}

int my_test_func(int i, int j) {
	return i + j + i * j;
}

int main(int argc, char *argv[])
{
	pid_t pid = getpid();
	printf("Victim running with PID %d\n", pid);
	printf("open %p, close %p\n", open, close);
	printf("my_test_func %p\n", my_test_func);

	int iterations = 10000; // number of iterations for the measurement
	int i;
	int res = 0;

	clock_t start, end;
	double elapsed_time;

	while (true) {
		start = clock();

		for (i = 0; i < iterations; i++) {
			// test_open();
			my_test_func(i , pid);
			// res = test_calc_fib((pid * 200) / (i + 1));
		}

		end = clock();
		elapsed_time = (end - start) / (double)CLOCKS_PER_SEC * 1000;
		printf("elapsed_time %fms\n", elapsed_time);
		printf("res %d\n", res);
		sleep(5);
	}
	return 0;
}
