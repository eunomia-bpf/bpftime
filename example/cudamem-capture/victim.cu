/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <stdio.h>
// 全局变量用于控制循环
extern "C" {
__device__ volatile int should_exit = 0;
__global__ void infinite_kernel()
{
	// 只让一个线程打印信息
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Kernel started\n");
	}

	// 计数器，用于周期性打印
	int counter = 0;

	while (true) {
		// 每个线程做一些简单计算以避免完全空循环
		float x = threadIdx.x + blockIdx.x;
		for (int i = 0; i < 1000; i++) {
			x = sinf(x) * x;
		}

		// 周期性打印状态（只用一个线程）
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			if (++counter % 1000000 == 0) {
				printf("Still running... counter=%d\n",
				       counter);
			}
		}

		// 防止编译器优化掉计算
		if (x == 0.0f) {
			should_exit = 1; // 永远不会发生
		}

		// 让出一些时间给其他线程
		__nanosleep(1000);
	}

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Kernel exiting\n");
	}
}
}
int main()
{
	// 设置设备
	cudaSetDevice(0);

	// 启动内核
	printf("Launching infinite kernel...\n");
	infinite_kernel<<<1, 32>>>();

	// 等待用户输入以结束程序
	printf("Press Enter to exit...\n");
	getchar();

	// 设置退出标志
	cudaMemcpyToSymbol(should_exit, &(int){ 1 }, sizeof(int));

	// 等待内核完成
	cudaDeviceSynchronize();
	printf("Program finished\n");

	return 0;
}
