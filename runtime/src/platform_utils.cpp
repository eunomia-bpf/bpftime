#include "platform_utils.hpp"

#if __linux__

int sched_getcpu() {
    return ::sched_getcpu();
}

#elif __APPLE__

int sched_getcpu() {
    int cpu = -1;
    size_t len = sizeof(cpu);

    if (sysctlbyname("hw.cpulocation", &cpu, &len, NULL, 0) == -1) {
        return -1;  // Handle error
    }
    return cpu;
}

int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask) {
    // Simulate affinity as macOS does not support setting/getting affinity
    // This is a stub function for compatibility
    (void)pid;
    (void)cpusetsize;
    CPU_ZERO(mask);
    CPU_SET(sched_getcpu(), mask);
    return 0;
}

int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
    // macOS does not support setting affinity, this is a no-op stub
    (void)pid;
    (void)cpusetsize;
    (void)mask;
    return 0;
}

#endif

int get_current_cpu() {
#if __linux__
    return sched_getcpu();
#elif __APPLE__
    return sched_getcpu();
#else
    return -1; // Unsupported platform
#endif
}
