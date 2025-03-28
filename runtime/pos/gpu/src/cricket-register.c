#include "cricket-register.h"
#include "log.h"

#include <stdio.h>

static bool cricket_register_rst_system_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                             uint32_t sm, uint32_t warp,
                                             uint32_t lane, void *register_data,
                                             uint32_t register_num)
{
    CUDBGResult res;
    uint32_t *regd = (uint32_t *)register_data;
    for (uint32_t i = 0; i < register_num; ++i) {
        // printf("setting R%u to %lx\n", i, regd[i]);
        res = cudbgAPI->writeRegister(dev, sm, warp, lane, i, regd[i]);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cricket_register_system_data \"%s\"",
                    cudbgGetErrorString(res));
            return false;
        }
    }
    return true;
}

static bool cricket_register_rst_cc_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                         uint32_t sm, uint32_t warp,
                                         uint32_t lane, void *register_data)
{
    CUDBGResult res;
    res = cudbgAPI->writeCCRegister(dev, sm, warp, lane,
                                    *(uint32_t *)register_data);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cricket_register_cc_data \"%s\"",
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

static bool cricket_register_rst_predicate_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                                uint32_t sm, uint32_t warp,
                                                uint32_t lane,
                                                void *register_data,
                                                uint32_t register_num)
{
    CUDBGResult res;
    res = cudbgAPI->writePredicates(dev, sm, warp, lane, register_num,
                                    (const uint32_t*)register_data);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cricket_register_predicate_data \"%s\"",
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

static bool cricket_register_ckp_system_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                             uint32_t sm, uint32_t warp,
                                             uint32_t lane, void *register_data,
                                             uint32_t register_num)
{
    CUDBGResult res;

    // uint32_t *regd = (uint32_t*)register_data;
    // for (uint32_t i = 0; i < register_num; ++i) {
    //    res = cudbgAPI->readRegister(dev, sm, warp, lane, i, regd+i);
    //    //printf("reading R%u: %lx\n", i, regd[i]);
    //    if (res != CUDBG_SUCCESS) {
    //        fprintf(stderr, "cricket_register_system_data \"%s\"\n",
    // cudbgGetErrorString(res));
    //        return false;
    //    }
    //}
    res = cudbgAPI->readRegisterRange(dev, sm, warp, lane, 0, register_num,
                                      (uint32_t*)register_data);
    // for (int i =0 ; i != register_num; ++i)
    //    printf("read R%d as %lx\n", i, ((uint32_t*)register_data)[i]);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cricket_register_system_data \"%s\"",
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

static bool cricket_register_ckp_cc_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                         uint32_t sm, uint32_t warp,
                                         uint32_t lane, void *register_data)
{
    CUDBGResult res;
    res = cudbgAPI->readCCRegister(dev, sm, warp, lane, (uint32_t*)register_data);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cricket_register_cc_data \"%s\"",
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

static bool cricket_register_ckp_predicate_data(CUDBGAPI cudbgAPI, uint32_t dev,
                                                uint32_t sm, uint32_t warp,
                                                uint32_t lane,
                                                void *register_data,
                                                uint32_t register_num)
{
    CUDBGResult res;
    res = cudbgAPI->readPredicates(dev, sm, warp, lane, register_num,
                                   (uint32_t*)register_data);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cricket_register_predicate_data \"%s\"",
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

size_t cricket_register_size(CricketDeviceProp *prop)
{
    return (prop->numRegisters + prop->numPredicates + 1) * sizeof(uint32_t);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
bool cricket_register_rst(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                          uint32_t warp, uint32_t lane, void *register_data,
                          CricketDeviceProp *prop)
{
    bool ret = false;
    void *dpos = register_data;
    ret = cricket_register_rst_system_data(cudbgAPI, dev, sm, warp, lane, dpos,
                                           prop->numRegisters);
    if (!ret)
        return ret;

    dpos += sizeof(uint32_t) * prop->numRegisters;
    ret = cricket_register_rst_cc_data(cudbgAPI, dev, sm, warp, lane, dpos);
    if (!ret)
        return ret;

    dpos += sizeof(uint32_t);
    ret = cricket_register_rst_predicate_data(cudbgAPI, dev, sm, warp, lane,
                                              dpos, prop->numPredicates);

    return ret;
}

bool cricket_register_ckp(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                          uint32_t warp, uint32_t lane, void *register_data,
                          CricketDeviceProp *prop)
{
    bool ret = false;
    void *dpos = register_data;
    ret = cricket_register_ckp_system_data(cudbgAPI, dev, sm, warp, lane, dpos,
                                           prop->numRegisters);
    if (!ret)
        return ret;

    dpos += sizeof(uint32_t) * prop->numRegisters;
    ret = cricket_register_ckp_cc_data(cudbgAPI, dev, sm, warp, lane, dpos);
    if (!ret)
        return ret;

    dpos += sizeof(uint32_t);
    ret = cricket_register_ckp_predicate_data(cudbgAPI, dev, sm, warp, lane,
                                              dpos, prop->numPredicates);

    return ret;
}
#pragma GCC diagnostic pop
