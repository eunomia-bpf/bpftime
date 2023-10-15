#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <unistd.h>

extern "C" int64_t simple_add(int64_t a, int64_t b)
{
	return a + b;
}

int main()
{
	while (true) {
		for (int i = 1; i <= 10; i++) {
			for (int j = 1; j <= 10; j++) {
				int32_t ret = simple_add(i, j);
				std::cout << i << " + " << j << " = " << ret
					  << std::endl;
				usleep(1000 * 500);
			}
		}
	}
	return 0;
}
