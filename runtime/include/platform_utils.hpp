#ifndef PLATFORM_UTIL_H
#define PLATFORM_UTIL_H
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string_view>

#if __linux__
#include <sched.h>
#elif __APPLE__
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

// The CPU id a BPF program sees is its leased slot: my_sched_getcpu() leases
// one for the outermost exec (per-CPU maps, bpf_get_smp_processor_id), its own
// CPU's or else the next free one, and perf output uses it, else the real CPU.
int my_sched_getcpu();
int bpftime_get_perf_output_cpu();
// The real CPU the thread runs on, cached for the outermost BPF exec.
int bpftime_get_current_cpu();
int bpftime_possible_cpus();
bool bpftime_enter_bpf_exec() noexcept;
void bpftime_leave_bpf_exec() noexcept;

namespace bpftime::rootless
{
// Use the tracefs mount path as the stable ID input. The debugfs alias must
// resolve to the same ID; otherwise the attach table cannot find that event.
inline int32_t synthetic_tracepoint_id(std::string_view id_path) noexcept
{
	constexpr std::string_view debugfs_prefix =
		"/sys/kernel/debug/tracing/events/";
	constexpr std::string_view tracefs_prefix =
		"/sys/kernel/tracing/events/";
	uint32_t hash = 2166136261U;
	const auto feed = [&](std::string_view path) {
		for (unsigned char ch : path) {
			hash ^= ch;
			hash *= 16777619U;
		}
	};
	if (id_path.starts_with(debugfs_prefix)) {
		feed(tracefs_prefix);
		id_path.remove_prefix(debugfs_prefix.size());
	}
	feed(id_path);
	return static_cast<int32_t>(1024U + hash % 100000000U);
}
} // namespace bpftime::rootless
#endif
