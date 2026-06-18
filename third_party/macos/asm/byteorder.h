#include <libkern/OSByteOrder.h>

#if defined(__LITTLE_ENDIAN__)
#define __LITTLE_ENDIAN_BITFIELD
#else
#define __BIG_ENDIAN_BITFIELD
#endif
