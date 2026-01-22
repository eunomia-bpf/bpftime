#ifndef BPFTIME_RUNTIME_BPF_MAP_GPU_NV_GPU_GDRCOPY_ABI_HPP
#define BPFTIME_RUNTIME_BPF_MAP_GPU_NV_GPU_GDRCOPY_ABI_HPP

#include <cstddef>
#include <cstdint>

#if __has_include(<gdrapi.h>)
#include <gdrapi.h>
#define BPFTIME_HAVE_GDRAPI_H 1
#endif

namespace bpftime
{
namespace gpu
{
namespace gdrcopy
{
namespace detail
{

#if defined(BPFTIME_HAVE_GDRAPI_H)
using gdr_t = ::gdr_t;
using gdr_mh_t = ::gdr_mh_t;
using gdr_info_t = ::gdr_info_t;
constexpr uint32_t kGdrPinDefaultFlags = ::GDR_PIN_FLAG_DEFAULT;
#else
// Minimal GDRCopy type declarations to allow dynamic loading via dlopen
struct gdr;
using gdr_t = gdr *;

struct gdr_mh_s {
	unsigned long h;
};
using gdr_mh_t = gdr_mh_s;

constexpr uint32_t kGdrPinDefaultFlags = 0;

enum gdr_mapping_type {
	GDR_MAPPING_TYPE_NONE = 0,
	GDR_MAPPING_TYPE_WC = 1,
	GDR_MAPPING_TYPE_CACHING = 2,
	GDR_MAPPING_TYPE_DEVICE = 3,
	GDR_MAPPING_TYPE_MAX
};

// Matches the ABI layout of gdrapi.h's gdr_info_t (used by gdr_get_info).
struct gdr_info {
	uint64_t va;
	uint64_t mapped_size;
	uint32_t page_size;
	uint64_t tm_cycles;
	uint32_t cycles_per_ms;
	unsigned mapped : 1;
	unsigned wc_mapping : 1;
	gdr_mapping_type mapping_type;
};
using gdr_info_t = gdr_info;
#endif

} // namespace detail
} // namespace gdrcopy
} // namespace gpu
} // namespace bpftime

#endif
