#include "platform_utils.hpp"
#include "bpftime_shm_internal.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
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
	if (shm.has_vcpu_locks()) {
		int vcpu = shm.try_lease_vcpu(cpu);
		if (vcpu >= 0) {
			current_bpf_cpu = vcpu;
			current_bpf_vcpu_leased = true;
			return vcpu;
		}
		current_bpf_cpu = BPF_CPU_LEASE_EXHAUSTED;
		return -1;
	}
	if (shm.get_open_type() != bpftime::shm_open_type::SHM_NO_CREATE) {
		current_bpf_cpu = BPF_CPU_LEASE_EXHAUSTED;
		return -1;
	}
	if (cpu >= 0)
		current_bpf_cpu = cpu;
	return cpu;
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

int bpftime_possible_cpus()
{
	return (int)std::clamp(sysconf(_SC_NPROCESSORS_CONF), 1L,
			       (long)bpftime::BPFTIME_MAX_CPU_COUNT);
}

int bpftime_vcpu_slots()
{
	const char *env = getenv("BPFTIME_VCPU_RATIO");
	const double ratio = env ? std::max(1.0, atof(env)) : 1.2;
	return (int)std::min(std::ceil(bpftime_possible_cpus() * ratio) + 1,
			     (double)bpftime::BPFTIME_MAX_CPU_COUNT);
}

int bpftime_percpu_columns()
{
	const int n = bpftime::shm_holder.global_shared_memory.vcpu_count();
	return n > 0 ? n : bpftime_possible_cpus();
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
