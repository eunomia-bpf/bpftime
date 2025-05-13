#include <Python.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <variant>

// 全局标志，用于控制Python程序的执行状态
std::atomic<bool> python_paused = false;

// 处理SIGUSR1信号，用于暂停Python执行
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

int main(int argc, char *argv[])
{
	// 示例Python代码 - 一个简单的计数循环
	// const char *python_code = "import time\n"
	// 			  "import sys\n"
	// 			  "\n"
	// 			  "print('Python program started')\n"
	// 			  "count = 0\n"
	// 			  "try:\n"
	// 			  "    while True:\n"
	// 			  "        count += 1\n"
	// 			  "        print(f'Counter: {count}')\n"
	// 			  "        sys.stdout.flush()\n"
	// 			  "        time.sleep(1)\n"
	// 			  "except KeyboardInterrupt:\n"
	// 			  "    print('Program interrupted')\n"
	// 			  "print('Python program finished')\n";

	std::string python_code;
	{
		std::ifstream ifs("cpu_pytorch.py", std::ios::ate);
		auto tail = ifs.tellg();
		ifs.seekg(0, std::ios::beg);
		// std::get()
		python_code.resize(tail);
		ifs.read(python_code.data(), tail);
	}

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

		// 运行Python代码
		PyRun_SimpleString(python_code.c_str());

		// 清理Python解释器
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

	return 0;
}
