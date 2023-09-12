#ifndef _RINGBUF_HELPERS_H
#define _RINGBUF_HELPERS_H
#define READ_ONCE_UL(x) (*(volatile unsigned long *)&x)
#define WRITE_ONCE_UL(x, v) (*(volatile unsigned long *)&x) = (v)
#define READ_ONCE_I(x) (*(volatile int *)&x)
#define WRITE_ONCE_I(x, v) (*(volatile int *)&x) = (v)

#define barrier() asm volatile("" ::: "memory")
#ifdef __x86_64__
#define smp_store_release_ul(p, v)                                             \
	do {                                                                   \
		barrier();                                                     \
		WRITE_ONCE_UL(*p, v);                                          \
	} while (0)

#define smp_load_acquire_ul(p)                                                 \
	({                                                                     \
		unsigned long ___p = READ_ONCE_UL(*p);                         \
		barrier();                                                     \
		___p;                                                          \
	})

#define smp_load_acquire_i(p)                                                  \
	({                                                                     \
		int ___p = READ_ONCE_I(*p);                                    \
		barrier();                                                     \
		___p;                                                          \
	})

#else
#error Only supports x86_64
#endif
#endif
