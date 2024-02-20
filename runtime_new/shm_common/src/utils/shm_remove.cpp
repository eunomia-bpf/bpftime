#include <utils/shm_remover.hpp>
#include <utils/global_shm.hpp>
#include <spdlog/spdlog.h>
#include <boost/interprocess/shared_memory_object.hpp>
using namespace bpftime::shm_common;
shm_remove::shm_remove()
{
	name = get_global_shm_name();
	boost::interprocess::shared_memory_object::remove(name.c_str());
}
shm_remove::shm_remove(const char *name) : name(name)
{
	boost::interprocess::shared_memory_object::remove(name);
}
shm_remove::~shm_remove()
{
	SPDLOG_INFO("Destroy shm {}", name);
	boost::interprocess::shared_memory_object::remove(name.c_str());
}
