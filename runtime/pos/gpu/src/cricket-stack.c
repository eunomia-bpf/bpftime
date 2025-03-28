
#include "cuda-tdep.h"
#include "objfiles.h"
#include "bfd.h"
#include "libbfd.h"
#include "elf-bfd.h"
#include "stdio.h"

#include "cricket-stack.h"

uint32_t cricket_stack_get_sp_regnum(void)
{
    struct gdbarch *cuda_gdbarch;
    cuda_gdbarch = cuda_get_gdbarch();
    return cuda_abi_sp_regnum(cuda_gdbarch);
}

bool cricket_stack_get_sp(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                          uint32_t warp, uint32_t lane, uint32_t *sp)
{
    CUDBGResult res;
    uint32_t sp_val;
    uint32_t sp_regnum = cricket_stack_get_sp_regnum();
    res = cudbgAPI->readRegister(dev, sm, warp, lane, sp_regnum, &sp_val);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-stack (%d) ERROR: %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }
    *sp = sp_val;
    return true;
}

bool cricket_stack_set_mem(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                           uint32_t warp, uint32_t lane, void *stack_mem,
                           uint32_t stack_loc, uint32_t stack_size)
{
    CUDBGResult res = cudbgAPI->writeLocalMemory(dev, sm, warp, lane, stack_loc,
                                                 stack_mem, stack_size);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-stack: (%d): %s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

bool cricket_stack_get_mem(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                           uint32_t warp, uint32_t lane, void *stack_mem,
                           uint32_t stack_size)
{
    uint32_t sp;
    CUDBGResult res;
    if (!cricket_stack_get_sp(cudbgAPI, dev, sm, warp, lane, &sp)) {
        return false;
    }

    res = cudbgAPI->readLocalMemory(dev, sm, warp, lane, sp, stack_mem,
                                    stack_size);
    if (res != CUDBG_SUCCESS) {
        printf("cricket-stack (%d):%s\n", __LINE__, cudbgGetErrorString(res));
        return false;
    }
    return true;
}

bool cricket_param_get_mem(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                           uint32_t warp, uint16_t param_addr, void *param_mem,
                           uint16_t param_size)
{
    CUDBGResult res;

    res = cudbgAPI->readParamMemory(dev, sm, warp, (uint64_t)param_addr,
                                    param_mem, (uint32_t)param_size);
    if (res != CUDBG_SUCCESS) {
        printf("cricket-param (%d):", __LINE__);
        return false;
    }
    return true;
}
