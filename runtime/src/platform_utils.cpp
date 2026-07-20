#include "platform_utils.hpp"
#include "spdlog/spdlog.h"
#include <cstdint>

namespace
{
thread_local int bpftime_current_bpf_cpu = -1;
thread_local uint32_t bpftime_current_bpf_cpu_depth = 0;
}

#if __linux__
#include <sched.h>

int my_sched_getcpu() {
    return ::sched_getcpu();
}

#elif __APPLE__
#include <sys/sysctl.h>
#include <pthread.h>

int my_sched_getcpu() {
    int cpu = -1;
    size_t len = sizeof(cpu);

    if (sysctlbyname("hw.cpulocation", &cpu, &len, NULL, 0) == -1) {
        SPDLOG_ERROR("Couldn't get cpu location for the system");
        return -1;  
    }
    return cpu;
}

int sched_getaffinity([[maybe_unused]] pid_t pid, [[maybe_unused]]size_t cpusetsize, cpu_set_t *mask) {
    CPU_ZERO(mask);
    CPU_SET(my_sched_getcpu(), mask);
    return 0;
}

int sched_setaffinity([[maybe_unused]]pid_t pid, [[maybe_unused]]size_t cpusetsize, [[maybe_unused]]const cpu_set_t *mask) {
    return 0;
}

#endif

namespace bpftime
{
int bpftime_get_current_cpu()
{
	if (bpftime_current_bpf_cpu >= 0) {
		return bpftime_current_bpf_cpu;
	}
	return my_sched_getcpu();
}

bpftime_bpf_cpu_guard::bpftime_bpf_cpu_guard()
{
	if (bpftime_current_bpf_cpu_depth == 0) {
		int cpu = my_sched_getcpu();
		if (cpu < 0) {
			SPDLOG_ERROR("sched_getcpu error");
			cpu = 0;
		}
		bpftime_current_bpf_cpu = cpu;
	}
	bpftime_current_bpf_cpu_depth++;
}

bpftime_bpf_cpu_guard::~bpftime_bpf_cpu_guard()
{
	if (bpftime_current_bpf_cpu_depth == 0) {
		return;
	}
	bpftime_current_bpf_cpu_depth--;
	if (bpftime_current_bpf_cpu_depth == 0) {
		bpftime_current_bpf_cpu = -1;
	}
}
} // namespace bpftime
