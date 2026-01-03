#pragma once

#include <cstdint>

#include "cuda.h"

namespace bpftime
{
namespace gpu
{
namespace gdrcopy
{

// Returns true iff GDRCopy was used to copy data from GPU to host.
// If it returns false, the caller should fall back to cuMemcpyDtoH.
bool copy_from_device_to_host_with_gdrcopy(const void *mapping_key,
					   const char *map_type_name,
					   CUdeviceptr device_base,
					   uint64_t total_buffer_size,
					   uint64_t copy_offset_bytes,
					   void *destination,
					   uint64_t copy_size_bytes,
					   uint64_t per_key_bytes);

void destroy_gdrcopy_mapping_for_owner(const void *mapping_key,
				       const char *map_type_name,
				       uint64_t total_buffer_size);

} // namespace gdrcopy
} // namespace gpu
} // namespace bpftime
