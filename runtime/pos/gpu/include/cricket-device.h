#ifndef _CRICKET_DEVICE_H_
#define _CRICKET_DEVICE_H_

#include "cricket-types.h"
#include <stddef.h>
#include "cudadebugger.h"

bool cricket_device_get_prop(CUDBGAPI cudbgAPI, uint32_t device_index,
                             CricketDeviceProp *prop);
bool cricket_device_get_num(CUDBGAPI cudbgAPI, uint32_t *dev_num);
void cricket_device_free_prop(CricketDeviceProp *prop);
void cricket_device_print_prop(CricketDeviceProp *prop);

#endif //_CRICKET_DEVICE_H_
