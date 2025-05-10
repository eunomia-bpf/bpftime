#ifndef _CUDATEST_H
#define _CUDATEST_H
struct map_key_type {
	unsigned long data[(1 << 20) / sizeof(unsigned long)];
	// short data[1];
};

#endif
