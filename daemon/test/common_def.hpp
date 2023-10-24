#ifndef _COMMON_DEF_HPP
#define _COMMON_DEF_HPP
#include <string>
#include <boost/interprocess/shared_memory_object.hpp>
struct shm_remove {
	std::string filename;
	shm_remove(const std::string &&filename) : filename(filename)
	{
		boost::interprocess::shared_memory_object::remove(
			filename.c_str());
	}

	~shm_remove()
	{
		boost::interprocess::shared_memory_object::remove(
			filename.c_str());
	}
};

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#endif
