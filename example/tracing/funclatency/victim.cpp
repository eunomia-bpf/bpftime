#include <chrono>
#include <iostream>
#include <ostream>
#include <random>
#include <thread>
extern "C" int plus(int a, int b)
{
	return a + b;
}

int main()
{
	std::mt19937 gen;
	gen.seed(std::random_device()());
	std::uniform_int_distribution<int> rand(1, 1e6);
	while (1) {
		int a = rand(gen);
		int b = rand(gen);
		int c = plus(a, b);
		std::cout << a << " + " << b << " = " << c << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return 0;
}
