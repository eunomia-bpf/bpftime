#ifndef _CRICKET_REGISTER_H_
#define _CRICKET_REGISTER_H_

#include <stddef.h>
#include "cudadebugger.h"
#include "cricket-device.h"

size_t cricket_register_size(CricketDeviceProp *prop);
bool cricket_register_ckp(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                          uint32_t warp, uint32_t lane, void *register_data,
                          CricketDeviceProp *prop);
bool cricket_register_rst(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                          uint32_t warp, uint32_t lane, void *register_data,
                          CricketDeviceProp *prop);

#endif //_CRICKET_REGISTER_H_
