#include "platform_utils.hpp"
#include "spdlog/spdlog.h"

namespace
{
constexpr int BPF_CPU_OUTSIDE_EXEC = -2;
constexpr int BPF_CPU_LAZY_UNSET = -1;
constinit thread_local int current_bpf_cpu = BPF_CPU_OUTSIDE_EXEC;

#if __linux__
int query_native_cpu()
{
	return ::sched_getcpu();
}
#elif __APPLE__
int query_native_cpu()
{
	size_t cpu = 0;
	// XNU has no sysctlbyname("hw.cpulocation"); pthread_cpu_number_np is
	// the SDK-supported way to query the logical CPU of the calling thread.
	if (pthread_cpu_number_np(&cpu) != 0) {
		SPDLOG_ERROR("Couldn't get cpu number for the current thread");
		return -1;
	}
	return static_cast<int>(cpu);
}
#endif
} // namespace

int my_sched_getcpu()
{
	if (current_bpf_cpu >= 0)
		return current_bpf_cpu;
	int cpu = query_native_cpu();
	if (current_bpf_cpu == BPF_CPU_LAZY_UNSET && cpu >= 0)
		current_bpf_cpu = cpu;
	return cpu;
}

bool bpftime_enter_bpf_exec() noexcept
{
	if (current_bpf_cpu != BPF_CPU_OUTSIDE_EXEC)
		return false;
	current_bpf_cpu = BPF_CPU_LAZY_UNSET;
	return true;
}

void bpftime_leave_bpf_exec() noexcept
{
	current_bpf_cpu = BPF_CPU_OUTSIDE_EXEC;
}

#if __APPLE__
int sched_getaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize, cpu_set_t *mask)
{
	CPU_ZERO(mask);
	int cpu = my_sched_getcpu();
	// A failed CPU query returns -1; never feed it to CPU_SET.
	if (cpu >= 0)
		CPU_SET(cpu, mask);
	return 0;
}

int sched_setaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize,
		      [[maybe_unused]] const cpu_set_t *mask)
{
	return 0;
}
#endif
