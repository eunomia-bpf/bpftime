#ifndef _MAP_COMMON_DEF_HPP
#define _MAP_COMMON_DEF_HPP
#include "spdlog/spdlog.h"
#include <cinttypes>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <functional>
#include <sched.h>
namespace bpftime
{

using bytes_vec_allocator = boost::interprocess::allocator<
	uint8_t, boost::interprocess::managed_shared_memory::segment_manager>;
using bytes_vec = boost::interprocess::vector<uint8_t, bytes_vec_allocator>;

template <class T> T ensure_on_current_cpu(std::function<T(int cpu)> func)
{
	cpu_set_t orig, set;
	CPU_ZERO(&orig);
	CPU_ZERO(&set);
	sched_getaffinity(0, sizeof(orig), &orig);
	int currcpu = sched_getcpu();
	CPU_SET(currcpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
	T ret = func(currcpu);
	sched_setaffinity(0, sizeof(orig), &orig);
	return ret;
}

} // namespace bpftime

#endif
