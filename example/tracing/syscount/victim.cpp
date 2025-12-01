#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <unistd.h>
int main()
{
	while (true) {
		std::cout << "Opening test.txt" << std::endl;

		int fd = open("test.txt", O_RDONLY | O_CREAT);
		assert(fd != -1);

		std::cout << "test.txt opened, fd=" << fd << std::endl;
		usleep(1000 * 300);
		std::cout << "Closing test.txt..." << std::endl;
		close(fd);
		std::cout << "test.txt closed" << std::endl;
	}
	return 0;
}
