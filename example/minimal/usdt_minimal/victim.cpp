#include <iostream>
#include <ostream>
#include <sys/sdt.h>  // provided by systemtap-sdt-devel package
#include <random>
#include <thread>
using namespace std::chrono_literals;
int main()
{
	std::mt19937 gen;
	gen.seed(std::random_device()());
	std::uniform_int_distribution<int> rand(1, 1e6);
	while (true) {
		int x = rand(gen);
		int y = rand(gen);
		int z = x + y;
		DTRACE_PROBE3(victim, probe1, x, y, z);
		std::cout << x << " + " << y << " = " << z << std::endl;
		int x1 = y;
		int y1 = x;
		DTRACE_PROBE3(victim, probe1, x1, y1, z);
		std::this_thread::sleep_for(1s);
	}
	return 0;
}
