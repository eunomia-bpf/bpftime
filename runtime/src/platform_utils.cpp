#include "platform_utils.hpp"
#include "spdlog/spdlog.h"

#if __linux__
#include <sched.h>

int my_sched_getcpu()
{
	return ::sched_getcpu();
}

#elif defined(__QNX__) || defined(BPFTIME_TARGET_QNX)
#include <sys/neutrino.h>
#include <sys/syspage.h>
#include <unistd.h>

int my_sched_getcpu()
{
	// QNX: current CPU for the calling thread (see SYSPAGE CPU info).
	// Fallback to 0 if the API is unavailable on a given SDP build.
#if defined(_NTO_TCTL_RUNMASK_GET_AND_SET)
	unsigned runmask = 0;
	if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, &runmask) != -1) {
		for (int i = 0; i < (int)sizeof(runmask) * 8; ++i) {
			if (runmask & (1u << i))
				return i;
		}
	}
#endif
	return 0;
}

int sched_getaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize, cpu_set_t *mask)
{
	CPU_ZERO(mask);
	CPU_SET(my_sched_getcpu(), mask);
	return 0;
}

int sched_setaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize,
		      [[maybe_unused]] const cpu_set_t *mask)
{
	return 0;
}

#elif __APPLE__
#include <sys/sysctl.h>
#include <pthread.h>

int my_sched_getcpu()
{
	int cpu = -1;
	size_t len = sizeof(cpu);

	if (sysctlbyname("hw.cpulocation", &cpu, &len, NULL, 0) == -1) {
		SPDLOG_ERROR("Couldn't get cpu location for the system");
		return -1;
	}
	return cpu;
}

int sched_getaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize, cpu_set_t *mask)
{
	CPU_ZERO(mask);
	CPU_SET(my_sched_getcpu(), mask);
	return 0;
}

int sched_setaffinity([[maybe_unused]] pid_t pid,
		      [[maybe_unused]] size_t cpusetsize,
		      [[maybe_unused]] const cpu_set_t *mask)
{
	return 0;
}

#endif
