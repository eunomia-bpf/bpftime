#ifndef _CRICKET_HEAP_H_
#define _CRICKET_HEAP_H_

#include "cudadebugger.h"
#include <stddef.h>

bool cricket_focus_host(bool batch_flag);
bool cricket_focus_kernel(bool batch_flag);
bool cricket_heap_memreg_size(void *addr, size_t *size);

#endif //_CRICKET_HEAP_H_
