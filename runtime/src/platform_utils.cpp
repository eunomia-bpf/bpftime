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
#include <pthread.h>

int bpftime_get_native_cpu() {
    size_t cpu = 0;
    // XNU has no sysctlbyname("hw.cpulocation"); pthread_cpu_number_np is the
    // SDK-supported way to query the logical CPU of the calling thread.
    if (pthread_cpu_number_np(&cpu) != 0) {
        SPDLOG_ERROR("Couldn't get cpu number for the current thread");
        return -1;
    }
    return static_cast<int>(cpu);
}

int my_sched_getcpu() {
    return current_bpf_cpu >= 0 ? current_bpf_cpu : bpftime_get_native_cpu();
}

int sched_getaffinity([[maybe_unused]] pid_t pid, [[maybe_unused]]size_t cpusetsize, cpu_set_t *mask) {
    CPU_ZERO(mask);
    int cpu = my_sched_getcpu();
    // A failed CPU query returns -1; never feed it to CPU_SET.
    if (cpu >= 0)
        CPU_SET(cpu, mask);
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
