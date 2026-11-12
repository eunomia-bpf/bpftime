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

#define ITER_COUNT 100000
uint64_t total_time = 0;
uint64_t count = 0;

void print_results()
{
	double avg = ((double)total_time / (double)count) / ITER_COUNT;
	std::cout << "Average time usage " << std::fixed << std::setprecision(5)
		  << avg << "ns, "
		  << " count " << count * ITER_COUNT << std::endl;
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
	print_results();
}

int main()
{
	puts("Done");
	test_syscall();
	return 0;
}
