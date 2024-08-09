#ifndef PLATFORM_UTIL_H
#define PLATFORM_UTIL_H
#include <cstdlib>
#include <functional>

#if __linux__
#include <sched.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <pthread.h>
    typedef int cpu_set_t;

    inline void CPU_ZERO(cpu_set_t *set) {
        *set = 0;
    }

    inline void CPU_SET(int cpu, cpu_set_t *set) {
        *set |= (1 << cpu);
    }

    inline int CPU_ISSET(int cpu, const cpu_set_t *set) {
        return (*set & (1 << cpu)) != 0;
    }
    int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask);
    int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask);
#else
    #error "Unsupported platform"
#endif

int my_sched_getcpu(); 
#endif