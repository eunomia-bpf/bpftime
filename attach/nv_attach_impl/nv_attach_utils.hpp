#ifndef _NV_ATTACH_UTILS_HPP
#define _NV_ATTACH_UTILS_HPP

#include <dlfcn.h>
namespace bpftime
{
namespace attach
{
extern "C" {
typedef struct {
	int magic;
	int version;
	const unsigned long long *data;
	void *filename_or_fatbins;

} __fatBinC_Wrapper_t;

}
template <class T>
static inline T try_get_original_func(const char *name, T &store)
{
	if (store == nullptr) {
		store = (T)dlsym(RTLD_NEXT, name);
	}
	return store;
}

} // namespace attach
} // namespace bpftime

#endif
