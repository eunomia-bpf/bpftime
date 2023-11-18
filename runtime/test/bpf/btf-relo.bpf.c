/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#define USE_NEW_VERSION

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push(__attribute__((preserve_access_index)),           \
				     apply_to = record)
#endif

typedef unsigned short uint16_t;

#define DISABLE_RELO 0

struct data_relo {
#if DISABLE_RELO
	int a;
#endif
	int b;
#if DISABLE_RELO
	int c;
#endif
	int d;
};

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

#define SEC(name)                                                              \
	_Pragma("GCC diagnostic push")                                         \
		_Pragma("GCC diagnostic ignored \"-Wignored-attributes\"")     \
			__attribute__((section(name), used))                   \
			_Pragma("GCC diagnostic pop")

SEC("prog")
int add_test(struct data_relo *d)
{
	return d->b + d->d;
}
