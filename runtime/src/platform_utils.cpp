#include "platform_utils.hpp"
#include "bpftime_shm_internal.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <unistd.h>

namespace
{
constexpr int BPF_CPU_LEASE_EXHAUSTED = -3;
constexpr int BPF_CPU_OUTSIDE_EXEC = -2;
constexpr int BPF_CPU_LAZY_UNSET = -1;
constinit thread_local int current_bpf_cpu = BPF_CPU_OUTSIDE_EXEC;
constinit thread_local int current_real_cpu = BPF_CPU_LAZY_UNSET;
constinit thread_local bool current_bpf_vcpu_leased = false;

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
	if (current_bpf_cpu == BPF_CPU_LEASE_EXHAUSTED)
		return -1;
	int cpu = bpftime_get_current_cpu();
	if (current_bpf_cpu != BPF_CPU_LAZY_UNSET)
		return cpu;
	auto &shm = bpftime::shm_holder.global_shared_memory;
	// No vCPU slots (no global shm, or one from an older version): per-CPU
	// maps have one column per real CPU.
	if (!shm.has_vcpu_locks()) {
		if (cpu >= 0)
			current_bpf_cpu = cpu;
		return cpu;
	}
	// A nested exec on this thread (a signal) sees no slot until the lease
	// is stored, so it never takes a second one.
	current_bpf_cpu = BPF_CPU_LEASE_EXHAUSTED;
	std::atomic_signal_fence(std::memory_order_seq_cst);
	int vcpu = shm.try_lease_vcpu(cpu);
	if (vcpu < 0)
		return -1;
	current_bpf_vcpu_leased = true;
	current_bpf_cpu = vcpu;
	return vcpu;
}

int bpftime_get_current_cpu()
{
	if (current_real_cpu >= 0)
		return current_real_cpu;
	int cpu = query_native_cpu();
	if (current_bpf_cpu != BPF_CPU_OUTSIDE_EXEC && cpu >= 0)
		current_real_cpu = cpu;
	return cpu;
}

int bpftime_get_perf_output_cpu()
{
	return current_bpf_cpu >= 0 ? current_bpf_cpu : bpftime_get_current_cpu();
}

int bpftime_possible_cpus()
{
	return (int)std::clamp(sysconf(_SC_NPROCESSORS_CONF), 1L,
			       (long)bpftime::BPFTIME_MAX_CPU_COUNT);
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
	if (current_bpf_vcpu_leased) {
		bpftime::shm_holder.global_shared_memory.release_vcpu(
			current_bpf_cpu);
		current_bpf_vcpu_leased = false;
	}
	current_real_cpu = BPF_CPU_LAZY_UNSET;
	current_bpf_cpu = BPF_CPU_OUTSIDE_EXEC;
}

#if __APPLE__
int sched_getaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize, cpu_set_t *mask)
{
	CPU_ZERO(mask);
	int cpu = bpftime_get_current_cpu();
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
