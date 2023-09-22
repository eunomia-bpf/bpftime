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
uint64_t total_time = 0;
uint64_t count = 0;

void sigint_handler(int)
{
	double avg = (double)total_time / (double)count;
	std::cout << "Average open time usage " << std::fixed
		  << std::setprecision(5) << avg << "ns, "
		  << " count " << count << std::endl;
	exit(0);
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
	puts("");
	while (1) {
		// DO NOT CHANGE THIS call to puts
		puts("Opening test.txt..");
		auto time_begin = std::chrono::high_resolution_clock::now();
		int fd = open("test.txt", O_CREAT | O_RDWR);
		auto time_end = std::chrono::high_resolution_clock::now();
		auto diff =
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				(time_end - time_begin));
		count += 1;
		total_time += diff.count();
		std::cout << "VICTIM: get fd " << fd << std::endl;
		usleep(500 * 1000);
		std::cout << "VICTIM: closing fd" << std::endl;
		close(fd);
	}
	return 0;
}
