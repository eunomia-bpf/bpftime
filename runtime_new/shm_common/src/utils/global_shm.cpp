#include <cstdlib>
#include <utils/global_shm.hpp>
namespace bpftime
{
namespace shm_common
{
const char *get_global_shm_name()
{
	const char *name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	if (name == nullptr) {
		return DEFAULT_GLOBAL_SHM_NAME;
	}
	return name;
}
} // namespace shm_common
} // namespace bpftime
