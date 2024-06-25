#include "platform_utils.hpp"
#include "spdlog/spdlog.h"

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

