#include "bpftime_shm.hpp"
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
	SPDLOG_INFO("Received sig {}", sig);
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

static void signal_handler_switch(int sig)
{
	// When SIGUSR2, do the switch
	int key = 1234;
	long value = 0;
	if (bpftime_map_lookup_elem(4, &key) == nullptr) {
		SPDLOG_INFO("Switching to GPU");
		bpftime_map_update_elem(4, &key, &value, 0);

	} else {
		SPDLOG_INFO("Switching to CPU");
		bpftime_map_delete_elem(4, &key);
	}
}

int main(int argc, char *argv[])
{
	auto executable_dir = std::filesystem::path(executable_pdir);
	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_OPEN_ONLY);
	{
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_handler = signal_handler_switch;
		sigaction(SIGUSR2, &sa, nullptr);
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

		// PyRun_SimpleString(python_code.c_str());
		auto python_file = executable_dir / "cpu_pytorch.py";
		auto fp = fopen(python_file.c_str(), "r");
		PyRun_SimpleFile(fp, python_file.c_str());
		Py_Finalize();

		printf("Python interpreter finished execution.\n");
	});

	std::thread timer_thread([&]() {
		int key = 1234;
		while (true) {
			// Running on CPU
			if (bpftime_map_lookup_elem(4, &key) == nullptr) {
				SPDLOG_INFO("Running on CPU");
				python_paused.store(false);
			} else {
				SPDLOG_INFO("Running on GPU");
				if (python_paused.load() == false) {
					SPDLOG_INFO("Stopping python thread..");
					pthread_kill(python_tid, SIGUSR1);
				}
			}
			std::this_thread::sleep_for(
				std::chrono::milliseconds(300));
		}
	});
	python_thd.join();
	timer_thread.join();
	return 0;
}
