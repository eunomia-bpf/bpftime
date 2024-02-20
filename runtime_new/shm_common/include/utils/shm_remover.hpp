#ifndef _BPFTIME_SHM_REMOVER_HPP
#define _BPFTIME_SHM_REMOVER_HPP
#include <string>
namespace bpftime
{
namespace shm_common
{
struct shm_remove {
	std::string name;
	shm_remove();
	shm_remove(const char *name);
	~shm_remove();
};
} // namespace shm_common
} // namespace bpftime

#endif
