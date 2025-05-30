#ifndef _NV_ATTACH_UTILS_HPP
#define _NV_ATTACH_UTILS_HPP

#include <dlfcn.h>
#include <string>
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

/**
 * @brief Get the default trampoline ptx object, used for helper operations,
 * which is generated from `test.cu`
 *
 * @return std::string
 */
std::string get_default_trampoline_ptx();
std::string patch_helper_names_and_header(std::string ptx_to_wrap);
std::string patch_main_from_func_to_entry(std::string);
std::string wrap_ptx_with_trampoline(std::string input);

} // namespace attach
} // namespace bpftime

#endif
