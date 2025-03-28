#include "defs.h"
#include "cricket-device.h"
#include <string.h>

bool cricket_device_get_num(CUDBGAPI cudbgAPI, uint32_t *dev_num)
{
    CUDBGResult res;
    res = cudbgAPI->getNumDevices(dev_num);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket_device_get_num (%d):%s\n", __LINE__,
                cudbgGetErrorString(res));
        return false;
    }
    return true;
}

bool cricket_device_get_prop(CUDBGAPI cudbgAPI, uint32_t device_index,
                             CricketDeviceProp *prop)
{
    CUDBGResult res = CUDBG_SUCCESS;
    const size_t STRLEN = 64;

    if (prop == NULL) {
        goto error_prop;
    }
    memset(prop, 0, sizeof(struct _CricketDeviceProp));
    prop->index = device_index;

    if ((prop->name = (char*)malloc(STRLEN)) == NULL) {
        goto error_name;
    }

    res = cudbgAPI->getDeviceName(device_index, prop->name, STRLEN);
    if (res != CUDBG_SUCCESS) {
        goto error_type;
    }

    if ((prop->type = (char*)malloc(STRLEN)) == NULL) {
        goto error_type;
    }

    res = cudbgAPI->getDeviceType(device_index, prop->type, STRLEN);
    if (res != CUDBG_SUCCESS) {
        goto error_smType;
    }

    if ((prop->smType = (char*)malloc(STRLEN)) == NULL) {
        goto error_smType;
    }

    res = cudbgAPI->getSmType(device_index, prop->smType, STRLEN);
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    res = cudbgAPI->getNumLanes(device_index, &(prop->numLanes));
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    res = cudbgAPI->getNumPredicates(device_index, &(prop->numPredicates));
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    res = cudbgAPI->getNumRegisters(device_index, &(prop->numRegisters));
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    res = cudbgAPI->getNumSMs(device_index, &(prop->numSMs));
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    res = cudbgAPI->getNumWarps(device_index, &(prop->numWarps));
    if (res != CUDBG_SUCCESS) {
        goto error;
    }

    return true;

error:
    free(prop->smType);
error_smType:
    free(prop->type);
error_type:
    free(prop->name);
error_name:
    free(prop);
error_prop:
    fprintf(stderr, "cricket_device_get_prop error: \"%s\"\n",
            cudbgGetErrorString(res));
    return false;
}

void cricket_device_free_prop(CricketDeviceProp *prop)
{
    free(prop->smType);
    free(prop->type);
    free(prop->name);
    free(prop);
}

void cricket_device_print_prop(CricketDeviceProp *prop)
{
    if (prop == NULL) {
        fprintf(stderr, "cricket_device_print_prop: prop is NULL\n");
        return;
    }

    printf("Device \"%s\":\n", prop->name);
    printf("\tindex: %u\n", prop->index);
    printf("\ttype: \"%s\"\n", prop->type);
    printf("\tSM type: \"%s\"\n", prop->smType);
    printf("\tlanes: %u\n", prop->numLanes);
    printf("\tpredicates %u\n", prop->numPredicates);
    printf("\tregisters: %u\n", prop->numRegisters);
    printf("\tSMs: %u\n", prop->numSMs);
    printf("\twarps: %u\n\n", prop->numWarps);
}
