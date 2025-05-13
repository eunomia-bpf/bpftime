#include <Python.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <variant>
#include <bpftime.hpp>
#include <libgen.h>
std::atomic<bool> python_paused = false;

void handle_signal(int sig)
{
	if (sig == SIGUSR1) {
		// 设置暂停标志
		python_paused.store(true);

		printf("Python interpreter paused. Waiting for resume signal...\n");

		// 等待恢复信号
		while (python_paused.load()) {
			// __asm__("pause;");
			std::this_thread::sleep_for(
				std::chrono::milliseconds(10));
		}

		printf("Python interpreter resumed execution.\n");
	}
}
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static const char *executable_pdir = TOSTRING(SOURCE_DIR);
int main(int argc, char *argv[])
{
	auto executable_dir = std::filesystem::path(executable_pdir);
	// SPDLOG_INFO("Changing work directory to ")
	// std::string python_code;
	// {
	// 	std::ifstream ifs(executable_dir / "cpu_pytorch.py",
	// 			  std::ios::ate);
	// 	auto tail = ifs.tellg();
	// 	ifs.seekg(0, std::ios::beg);
	// 	// std::get()
	// 	python_code.resize(tail);
	// 	ifs.read(python_code.data(), tail);
	// }

	pthread_t python_tid;

	std::thread python_thd([&]() {
		python_tid = pthread_self();
		// 初始化Python解释器
		Py_Initialize();

		// 设置信号处理器
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_handler = handle_signal;
		sigaction(SIGUSR1, &sa, NULL);

		printf("Python interpreter started. Running code...\n");

		// PyRun_SimpleString(python_code.c_str());
		auto python_file = executable_dir / "cpu_pytorch.py";
		auto fp = fopen(python_file.c_str(), "r");
		PyRun_SimpleFile(fp, python_file.c_str());
		Py_Finalize();

		printf("Python interpreter finished execution.\n");
	});
	
	std::thread timer_thread([&]() {
		// 等待10秒后暂停Python执行
		sleep(10);

		// 发送暂停信号
		pthread_kill(python_tid, SIGUSR1);

		// 等待1分钟
		printf("Waiting for 10 seconds before resuming...\n");
		sleep(10);

		// 恢复Python执行
		python_paused.store(false);
	});
	python_thd.join();
	timer_thread.join();
	return 0;
}
