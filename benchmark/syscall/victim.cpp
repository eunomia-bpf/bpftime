#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cinttypes>
#include <iomanip>
#include <signal.h>

#define ITER_COUNT 100000
uint64_t total_time = 0;
uint64_t count = 0;

void sigint_handler(int signum)
{
	double avg = ((double)total_time / (double)count) / ITER_COUNT;
	std::cout << "Average time usage " << std::fixed << std::setprecision(5)
		  << avg << "ns, "
		  << " count " << count * ITER_COUNT << std::endl;
	if (signum != 0)
		exit(0);
}

void test_syscall()
{
	int fd = open("/dev/null", O_CREAT | O_RDWR);
	while (count < 10) {
		printf("Iteration %lu\n", count);
		// DO NOT CHANGE THIS call to puts
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < ITER_COUNT; i++) {
			write(fd, "hello", 5);
		}
		auto time_end = std::chrono::high_resolution_clock::now();
		auto diff =
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				(time_end - time_begin));
		count += 1;
		total_time += diff.count();
	}
	printf("Total time: %" PRIu64 "\n", total_time);
	close(fd);
	sigint_handler(0);
}

int main()
{
	{
		struct sigaction sa;

		sa.sa_handler = sigint_handler;
		sa.sa_flags = SA_RESTART;
		if (sigaction(SIGINT, &sa, NULL) == -1) {
			std::cerr << "Failed to set signal handler"
				  << std::endl;
			return 1;
		}
	}
	puts("Done");
	test_syscall();
	return 0;
}
