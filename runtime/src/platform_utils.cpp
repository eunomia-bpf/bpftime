#include "platform_utils.hpp"
#include "spdlog/spdlog.h"

static thread_local int current_bpf_cpu = -1;

#if __linux__
#include <sched.h>

int my_sched_getcpu() {
    return current_bpf_cpu >= 0 ? current_bpf_cpu : ::sched_getcpu();
}

int bpftime_get_native_cpu() {
    return ::sched_getcpu();
}

#elif __APPLE__
#include <sys/sysctl.h>
#include <pthread.h>

int bpftime_get_native_cpu() {
    int cpu = -1;
    size_t len = sizeof(cpu);

    if (sysctlbyname("hw.cpulocation", &cpu, &len, NULL, 0) == -1) {
        SPDLOG_ERROR("Couldn't get cpu location for the system");
        return -1;  
    }
    return cpu;
}

int my_sched_getcpu() {
    return current_bpf_cpu >= 0 ? current_bpf_cpu : bpftime_get_native_cpu();
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

int bpftime_get_current_bpf_cpu() {
    return current_bpf_cpu;
}

void bpftime_set_current_bpf_cpu(int cpu) {
    current_bpf_cpu = cpu;
}
