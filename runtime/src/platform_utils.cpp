#include "platform_utils.hpp"

#if defined(__linux__)

int sched_getcpu() {
    return ::sched_getcpu();
}

#elif defined(__APPLE__) && defined(__MACH__)

int sched_getcpu() {
    int cpu = -1;
    size_t len = sizeof(cpu);
    int mib[2] = { CTL_HW, HW_CPUID };

    if (sysctl(mib, 2, &cpu, &len, NULL, 0) == -1) {
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
