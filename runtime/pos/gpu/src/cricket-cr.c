#include "cuda-tdep.h"
#include <stdio.h>
#include <sys/time.h>
#include "log.h"
#include "cricket-cr.h"
#include "cricket-elf.h"
#include "cricket-file.h"
#include "cricket-heap.h"
#include "cricket-register.h"
#include "cricket-stack.h"

bool cricket_cr_function_name(uint64_t pc, const char **name)
{
    CUDBGResult res;
    const char *function_name;

    function_name = cuda_find_function_name_from_pc(pc, false);
    *name = function_name;
    return function_name != NULL;

cuda_error:
    fprintf(stderr, "Cuda Error: \"%s\"\n", cudbgGetErrorString(res));
    return false;
}

bool cricket_cr_sm_broken(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm)
{
    uint64_t warp_mask;
    uint64_t warp_mask_broken;
    CUDBGResult res;
    res = cudbgAPI->readValidWarps(dev, sm, &warp_mask);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda api error");
        goto cuda_error;
    }
    res = cudbgAPI->readBrokenWarps(dev, sm, &warp_mask_broken);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda api error");
        goto cuda_error;
    }
    if (warp_mask != warp_mask_broken) {
        return false;
    }
    return true;
cuda_error:
    LOGE(LOG_ERROR, "Cuda Error: \"%s\"", cudbgGetErrorString(res));
    return false;
}
bool cricket_cr_kernel_name(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                            uint32_t wp, const char **name)
{
    CUDBGResult res;
    uint64_t grid_id;
    CUDBGGridInfo info;
    const char *kernel_name;

    res = cudbgAPI->readGridId(dev, sm, wp, &grid_id);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda api error");
        goto cuda_error;
    }

    res = cudbgAPI->getGridInfo(dev, grid_id, &info);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda api error");
        goto cuda_error;
    }

    kernel_name = cuda_find_function_name_from_pc(info.functionEntry, false);
    *name = kernel_name;
    return kernel_name != NULL;

cuda_error:
    LOGE(LOG_ERROR, "Cuda Error: \"%s\"", cudbgGetErrorString(res));
    return false;
}

static bool cricket_cr_gen_suffix(char **suffix, cricketWarpInfo *wi,
                                  uint32_t lane)
{
    size_t ret;
    if (lane == CRICKET_CR_NOLANE) {
        ret = asprintf(suffix, "-D%uS%uW%u", wi->dev, wi->sm, wi->warp);
    } else {
        ret =
            asprintf(suffix, "-D%uS%uW%uL%u", wi->dev, wi->sm, wi->warp, lane);
    }
    if (ret < 0) {
        LOGE(LOG_ERROR, "memory allocation failed");
        return false;
    }
    return true;
}

#define CRICKET_PROFILE 1
bool cricket_cr_rst_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir)
{
    CUDBGResult res;
    size_t register_size;
    void *reg_mem = NULL;
    void *stack_mem = NULL;
    char *suffix;
    bool ret = false;
    uint32_t stack_loc;
#ifdef CRICKET_PROFILE
    struct timeval b, c, d, e, g;
    double ct, dt, et, gt, comt;
#endif

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&b, NULL);
#endif
    register_size = cricket_register_size(wi->dev_prop);
    if ((reg_mem = malloc(register_size)) == NULL) {
        LOGE(LOG_ERROR,
                "error during memory allocation of size %lu",
                register_size);
        goto cleanup;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_REGISTERS, suffix, reg_mem,
                               register_size)) {
        LOGE(LOG_ERROR, "error while setting registers");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("register-data: ");
        for (int i = 0; i != register_size / sizeof(uint32_t); ++i) {
            printf("%08x ", ((uint32_t *)reg_mem)[i]);
        }
        printf("\n");
    }
    stack_loc = ((uint32_t *)reg_mem)[cricket_stack_get_sp_regnum()];

#ifdef CRICKET_PROFILE
    gettimeofday(&c, NULL);
#endif

    if (!cricket_register_rst(cudbgAPI, wi->dev, wi->sm, wi->warp, lane,
                              reg_mem, wi->dev_prop)) {
        LOGE(LOG_ERROR, "cricket-cr: error setting register data");
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&d, NULL);
#endif

    if ((stack_mem = malloc(wi->stack_size)) == NULL) {
        LOGE(LOG_ERROR, "error during memory allocation of size %lu",
                register_size);
        goto cleanup;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_STACK, suffix, stack_mem,
                               wi->stack_size)) {
        LOGE(LOG_ERROR, "error while setting stack memory");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("stack-mem: ");
        for (int i = 0; i != wi->stack_size; ++i) {
            printf("%02x ", ((uint8_t *)stack_mem)[i]);
        }
        printf("\n");
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&e, NULL);
#endif

    if (!cricket_stack_set_mem(cudbgAPI, wi->dev, wi->sm, wi->warp, lane,
                               stack_mem, stack_loc, wi->stack_size)) {
        LOGE(LOG_ERROR, "cricket-cr: error while retrieving stack memory");
        goto cleanup;
    }

#ifdef CRICKET_PROFILE
    gettimeofday(&g, NULL);
    ct = ((double)((c.tv_sec * 1000000 + c.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
                1000000.;
    dt = ((double)((d.tv_sec * 1000000 + d.tv_usec) -
                          (c.tv_sec * 1000000 + c.tv_usec))) /
                1000000.;
    et = ((double)((e.tv_sec * 1000000 + e.tv_usec) -
                          (d.tv_sec * 1000000 + d.tv_usec))) /
                1000000.;
    gt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                          (e.tv_sec * 1000000 + e.tv_usec))) /
                1000000.;
    comt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                            (b.tv_sec * 1000000 + b.tv_usec))) /
                  1000000.;
    LOG(LOG_DEBUG, "lane time:\n\t\tPROFILE readreg: %f s\n\t\tPROFILE setreg: %f "
           "s\n\t\tPROFILE readstack: %f s\n\t\tPROFILE setstack: %f "
           "s\n\t\tPROFILE lanecomplete: %f s",
           ct, dt, et, gt, comt);
#endif
    ret = true;
cleanup:
    free(reg_mem);
    free(stack_mem);
    free(suffix);
    return ret;
}

#define CRICKET_INSTR_SSY_PREFIX 0xe2900000
bool cricket_cr_ckp_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                        const char *ckp_dir)
{
    uint64_t pc;
    uint64_t virt_pc;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_syn_pc;
    uint32_t syn_pc = 0;
    uint64_t sswarps;
    CUDBGResult res;

    res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
        return false;
    }
    res = cudbgAPI->readVirtualPC(wi->dev, wi->sm, wi->warp, lane, &virt_pc);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
        return false;
    }
    for (offset = 0L; offset <= pc; offset += 0x8) {
        res = cudbgAPI->readCodeMemory(wi->dev, virt_pc - offset, &cur_instr,
                                       sizeof(uint64_t));
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
            return false;
        }
        LOGE(LOG_DEBUG, "instr: 0x%lx", cur_instr);
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_INSTR_SSY_PREFIX) {
            rel_syn_pc = ((cur_instr >> 20) & 0x000ffffffffL);
            LOGE(LOG_DEBUG, "rel_syn_pc: %x", rel_syn_pc);
            syn_pc = (pc - offset) + rel_syn_pc + 0x8;
            LOGE(LOG_DEBUG, "syn_pc: %lx, is bigger: %d", syn_pc, syn_pc > pc);

            break;
        }
    }

    while (syn_pc > pc) {
        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, 1, &sswarps);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
        }
        res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
        }
    }
    LOGE(LOG_DEBUG, "new pc: %lx", pc);
    return true;
}

void cricket_cr_free_callstack(cricket_callstack *callstack)
{
    if (callstack == NULL)
        return;

    free(callstack->function_names);
    callstack->function_names = NULL;
    free(callstack->_packed_ptr);
    callstack->_packed_ptr = NULL;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
bool cricket_cr_read_pc(cricketWarpInfo *wi, uint32_t lane, const char *ckp_dir,
                        cricket_callstack *callstack)
{
    CUDBGResult res;
    void *packed = NULL;
    size_t packed_size;
    uint32_t callstack_size;
    uint32_t valid_lanes;
    uint32_t active_lanes;
    char *suffix = NULL;
    bool ret = false;
    cricket_pc_data *pc_data;
    const char **function_names = NULL;
    size_t offset = 3 * sizeof(uint32_t);
    size_t i;

    if (callstack == NULL) {
        return false;
    }

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

    if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_PC, suffix, &packed, 0,
                                    &packed_size)) {
        LOGE(LOG_ERROR, "error while reading pc memory");
        goto cleanup;
    }

    if (packed_size < offset) {
        LOGE(LOG_ERROR, "pc checkpoint file is corrupt: no "
                        "callstack_size");
        goto cleanup;
    }

    valid_lanes = *(uint32_t *)(packed);
    active_lanes = *(uint32_t *)(packed + sizeof(uint32_t));
    callstack_size = *(uint32_t *)(packed + 2 * sizeof(uint32_t));

    LOGE(LOG_DEBUG, "valid_lanes: %x, active_lanes: %x, callstack_size: %x",
           valid_lanes, active_lanes, callstack_size);

    offset += callstack_size * sizeof(cricket_pc_data);
    if (packed_size < offset) {
        LOGE(LOG_ERROR, "pc checkpoint file is corrupt: too few "
                        "pc_data entries");
        goto cleanup;
    }

    pc_data = (cricket_pc_data *)(packed + 3 * sizeof(uint32_t));

    if (packed_size < offset + pc_data[callstack_size - 1].str_offset + 1) {
        LOGE(LOG_ERROR, "pc checkpoint file is corrupt: string "
                        "data to short");
        goto cleanup;
    }

    if ((function_names = (const char**)malloc(callstack_size * sizeof(char *))) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        goto cleanup;
    }

    for (i = 0; i < callstack_size; ++i) {
        if (packed_size < offset + pc_data[i].str_offset) {
            LOGE(LOG_ERROR, "pc checkpoint file is corrupt: string "
                            "data to short");
            goto cleanup;
        }

        function_names[i] = (const char*)(packed + offset + pc_data[i].str_offset);
    }

    IFLOG(LOG_DBG(3)) {
        LOGE(LOG_DEBUG, "callstack_size: %u", callstack_size);
        for (i = 0; i < callstack_size; ++i) {
            printf("\trelative: %lx, absolute: %lx, str_offset: %lx, function: %s\n",
                   pc_data[i].relative, pc_data[i].absolute, pc_data[i].str_offset,
                   function_names[i]);
        }
    }

    callstack->valid_lanes = valid_lanes;
    callstack->active_lanes = active_lanes;
    callstack->callstack_size = callstack_size;
    callstack->pc = pc_data;
    callstack->function_names = function_names;
    free(suffix);
    return true;
cleanup:
    free(packed);
    free(function_names);
    free(suffix);
    return false;
}
#pragma GCC diagnostic pop

// One specifc warp
bool cricket_cr_rst_subcall(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                            cricket_callstack *callstack, int c_level,
                            cricket_jmptable_entry *cal, size_t cal_num,
                            uint64_t *cur_address)
{
    uint64_t jmptbl_address;
    uint32_t lanemask;
    bool ret = false;
    uint64_t sswarps;
    CUDBGResult res;

    if (!cricket_elf_get_jmptable_addr(
             cal, cal_num, callstack->pc[c_level].relative, &jmptbl_address)) {
        LOGE(LOG_ERROR, "error getting jmptable adress");
        goto error;
    }

    res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &lanemask);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error");
        goto error;
    }

    for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
        if (lanemask & (1LU << lane)) {
            res = cudbgAPI->writeRegister(
                wi->dev, wi->sm, wi->warp, lane, CRICKET_JMX_ADDR_REG,
                ((uint32_t)(jmptbl_address - *cur_address - 0x8)));
            if (res != CUDBG_SUCCESS) {
                LOGE(LOG_ERROR, "cuda error: %s",
                        cudbgGetErrorString(res));
                goto error;
            }
        }
    }

    res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, 1, &sswarps);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error");
        goto error;
    }

    res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, 1, &sswarps);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error");
        goto error;
    }
    *cur_address = jmptbl_address + 0x8;

    ret = true;
error:
    return ret;
}
// One specifc warp
bool cricket_cr_rst_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                        cricket_callstack *callstack, int c_level,
                        cricket_jmptable_entry *ssy, size_t ssy_num,
                        uint64_t *cur_address)
{
    uint64_t relative_ssy;
    uint64_t jmptbl_address;
    uint32_t lanemask;
    bool ret = false;
    uint64_t sswarps;
    uint64_t cmp_address;
    CUDBGResult res;

    if (!cricket_elf_pc_info(callstack->function_names[c_level],
                             callstack->pc[c_level].relative, &relative_ssy,
                             NULL)) {
        LOGE(LOG_ERROR, "cricket-restore: getting ssy point failed");
        goto error;
    }
    cmp_address = callstack->pc[c_level].relative;
    if (relative_ssy % (4 * 8) == 0) {
        cmp_address -= 0x8;
    }

    if (relative_ssy >= cmp_address) {
        if (!cricket_elf_get_jmptable_addr(ssy, ssy_num, relative_ssy,
                                           &jmptbl_address)) {
            LOGE(LOG_ERROR, "error getting jmptable adress");
            goto error;
        }

        res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &lanemask);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error");
            goto error;
        }

        for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
            if (lanemask & (1LU << lane)) {
                res = cudbgAPI->writeRegister(
                    wi->dev, wi->sm, wi->warp, lane, CRICKET_JMX_ADDR_REG,
                    ((uint32_t)(jmptbl_address - *cur_address - 0x8)));
                if (res != CUDBG_SUCCESS) {
                    LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
                    goto error;
                }
            }
        }

        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, 1, &sswarps);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
            goto error;
        }

        res = cudbgAPI->singleStepWarp(wi->dev, wi->sm, wi->warp, 1, &sswarps);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
            goto error;
        }
        *cur_address = jmptbl_address + 0x8;
        if (*cur_address % (4 * 8) == 0) {
            *cur_address += 0x8;
        }
        LOG(LOG_INFO, "restored ssy");
    }
    ret = true;
error:
    return ret;
}

bool cricket_cr_rst_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_param, cricket_callstack *callstack)
{
    CUDBGResult res;
    bool ret = false;
    uint32_t lane;
    uint64_t address;
    uint32_t predicate;

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
    } else {
        lane = lane_param;
    }
    if (!(callstack->valid_lanes & (1 << lane))) {
        LOGE(LOG_ERROR, "lane %d is not valid", lane);
    }

    if (callstack->callstack_size == 1) {
        address = callstack->pc[0].relative - 0x20;
    } else if (callstack->callstack_size == 2) {
        if (callstack->active_lanes != callstack->valid_lanes) {
            LOGE(LOG_ERROR, "cricket-cr: divergent threads in call levels > 1 "
                            "are not allowed!");
            return false;
        }
        address = callstack->pc[0].absolute;

        /*if (callstack->active_lanes & (1<<lane)) {
            address = callstack->pc[0].absolute;
        } else {
            address =
        callstack->pc[0].absolute-callstack->pc[0].relative+0x1150;
        }*/
    } else {
        LOGE(LOG_ERROR, "cricket-cr: callstacks greater than 2 cannot be "
                        "restored");
        return false;
    }

    if (address > (1ULL << 32)) {
        LOGE(LOG_ERROR, "cricket-cr: pc value is too large to be restored!");
        goto cleanup;
    }

    if (callstack->active_lanes & (1 << lane)) {
        predicate = 0;
    } else {
        predicate = 1;
    }

    res = cudbgAPI->writePredicates(wi->dev, wi->sm, wi->warp, lane, 1,
                                    &predicate);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        return false;
    }

    do {
        res = cudbgAPI->writeRegister(wi->dev, wi->sm, wi->warp, lane,
                                    CRICKET_JMX_ADDR_REG, ((uint32_t)address));

        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "D%uS%uW%uL%u: %s, retrying...",
                    wi->dev, wi->sm, wi->warp, lane,
                    cudbgGetErrorString(res));
        }
    } while (res != CUDBG_SUCCESS);

    /*res = cudbgAPI->writeRegister(wi->dev, wi->sm, wi->warp, lane,
    CRICKET_JMX_ADDR_REG+1, (uint32_t)0x0);
    if (res != CUDBG_SUCCESS) {
        fprintf(stderr, "cricket-cr (%d): %s\n", __LINE__,
    cudbgGetErrorString(res));
        goto cleanup;
    }*/

    ret = true;
cleanup:
    return ret;
}

bool cricket_cr_callstack(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                          uint32_t lane_param, cricket_callstack *callstack)
{
    uint32_t base_addr;
    uint64_t call_instr;
    cricket_pc_data *pc_data = NULL;
    const char **function_names = NULL;
    uint32_t i;
    uint32_t callstack_size;
    uint32_t active_lanes;
    uint32_t valid_lanes;
    size_t str_offset = 0;
    bool ret = false;
    uint32_t lane;
    size_t offset = 0;
    CUDBGResult res;

    if (callstack == NULL)
        return false;

    res = cudbgAPI->readValidLanes(wi->dev, wi->sm, wi->warp, &valid_lanes);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        return false;
    }

    res = cudbgAPI->readActiveLanes(wi->dev, wi->sm, wi->warp, &active_lanes);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        return false;
    }

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
        for (uint32_t lane = 0; lane != wi->dev_prop->numLanes; lane++) {
            if (active_lanes & (1LU << lane)) {
                lane = i;
                break;
            }
        }
    } else {
        lane = lane_param;
    }

    res = cudbgAPI->readCallDepth(wi->dev, wi->sm, wi->warp, lane,
                                  &callstack_size);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        return false;
    }
    callstack_size++;

    if ((function_names = (const char**)malloc(callstack_size * sizeof(char *))) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return false;
    }
    if ((pc_data = (cricket_pc_data*)malloc(callstack_size * sizeof(cricket_pc_data))) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        goto cleanup;
    }

    res = cudbgAPI->readPC(wi->dev, wi->sm, wi->warp, lane, &pc_data[0].relative);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }

    res = cudbgAPI->readVirtualPC(wi->dev, wi->sm, wi->warp, lane,
                                  &pc_data[0].virt);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }

    if (!cricket_cr_function_name(pc_data[0].virt, &function_names[0])) {
        LOGE(LOG_ERROR, "cricket-cr: error getting function name");
        goto cleanup;
    }
    LOGE(LOG_DEBUG, "relative %lx, virtual %lx", pc_data[0].relative, pc_data[0].virt);

    pc_data[0].str_offset = 0;
    str_offset = strlen(function_names[0]) + 1;

    for (i = 1; i < callstack_size; ++i) {
        res = cudbgAPI->readReturnAddress(wi->dev, wi->sm, wi->warp, lane,
                                          i - 1, &pc_data[i].relative);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cricket error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        res = cudbgAPI->readVirtualReturnAddress(wi->dev, wi->sm, wi->warp,
                                                 lane, i - 1, &pc_data[i].virt);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cricket error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        if (!cricket_cr_function_name(pc_data[i].virt, &function_names[i])) {
            LOGE(LOG_ERROR, "error getting function name");
            goto cleanup;
        }
        pc_data[i].str_offset = str_offset;
        str_offset += strlen(function_names[i]) + 1;

        res = cudbgAPI->readCodeMemory(wi->dev,
                                       pc_data[i].virt - sizeof(uint64_t),
                                       &call_instr, sizeof(uint64_t));
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        base_addr = (call_instr >> 20);

        pc_data[i - 1].absolute = base_addr + pc_data[i - 1].relative;
    }
    pc_data[callstack_size - 1].absolute = 0;

    callstack->active_lanes = active_lanes;
    callstack->valid_lanes = valid_lanes;
    callstack->callstack_size = callstack_size;
    callstack->pc = pc_data;
    callstack->function_names = function_names;
    callstack->_packed_ptr = NULL;

    ret = true;
    return ret;

cleanup:
    free(function_names);
    free(pc_data);
    return ret;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
bool cricket_cr_ckp_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_param, const char *ckp_dir,
                       cricket_callstack *callstack)
{
    char *suffix;
    void *packed = NULL;
    size_t packed_size;
    bool ret = false;
    size_t str_size;
    size_t offset = 0;
    uint32_t lane;
    size_t i;
    CUDBGResult res;
    if (callstack == NULL || wi == NULL || ckp_dir == NULL)
        return false;

    if (lane_param == CRICKET_CR_NOLANE) {
        lane = 0;
    } else {
        lane = lane_param;
    }
    str_size =
        callstack->pc[callstack->callstack_size - 1].str_offset +
        strlen(callstack->function_names[callstack->callstack_size - 1]) + 1;

    packed_size = 3 * sizeof(uint32_t) +
                  callstack->callstack_size * sizeof(cricket_pc_data) +
                  str_size;
    if ((packed = malloc(packed_size)) == NULL) {
        goto cleanup;
    }

    memcpy(packed, &callstack->valid_lanes, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, &callstack->active_lanes, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, &callstack->callstack_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(packed + offset, callstack->pc,
           callstack->callstack_size * sizeof(cricket_pc_data));
    offset += callstack->callstack_size * sizeof(cricket_pc_data);
    for (i = 0; i < callstack->callstack_size; ++i) {
        strcpy((char*)(packed + offset + callstack->pc[i].str_offset),
               callstack->function_names[i]);
    }

    IFLOG(LOG_DBG(3)) {
        LOGE(LOG_DEBUG, "callstack_size: %u", callstack->callstack_size);
        for (i = 0; i < callstack->callstack_size; ++i) {
            printf("relative: %lx, absolute: %lx, str_offset: %lx, function: %s\n",
                   callstack->pc[i].relative, callstack->pc[i].absolute,
                   callstack->pc[i].str_offset, callstack->function_names[i]);
        }
    }

    if (!cricket_cr_gen_suffix(&suffix, wi, lane_param)) {
        goto cleanup;
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PC, suffix, packed,
                                packed_size)) {
        fprintf(stderr, "cricket-cr: error writing pc\n");
        goto cleanup;
    }

    ret = true;

cleanup:
    free(packed);
    free(suffix);
    return ret;
}
#pragma GCC diagnostic pop

bool cricket_cr_make_checkpointable(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                                    cricket_function_info *fi, size_t fi_num,
                                    cricket_callstack *callstack)
{
    CUDBGResult res;
    uint64_t rel_ssy, rel_pbk;
    cricket_function_info *the_fi = NULL;
    bool ret = false;
    size_t i;
    bool joined = false;
    bool stepup = false;
    size_t callstack_bku = callstack->callstack_size;
    for (i = callstack->callstack_size - 1; i + 1 > 0; --i) {
        if (!cricket_elf_get_fun_info(fi, fi_num, callstack->function_names[i],
                                      &the_fi)) {
            LOGE(LOG_ERROR, "failed to get fun_info");
            goto cleanup;
        }
        if (the_fi == NULL) {
            LOGE(LOG_ERROR, "no info for function %s available",
                    callstack->function_names[i]);
            goto cleanup;
        }
        LOGE(LOG_DEBUG, "function \"%s\" has %sroom (%lu slots)",
               callstack->function_names[i], (the_fi->has_room ? "" : "no "),
               the_fi->room);
        if (i == callstack->callstack_size - 1 && the_fi->room == 0) {
            LOGE(LOG_ERROR, "There is no room in the top level "
                            "function (i.e. the kernel). This kernel can thus "
                            "never be restored!");
            goto cleanup;
        }
        if (!the_fi->has_room) {
            LOGE(LOG_WARNING, "function %s does not have enough room. Subcalls and "
                   "synchronization points in this function cannot be "
                   "estored",
                   callstack->function_names[i]);
            if (i > 1) {
                LOGE(LOG_WARNING, "no room in %s. going up to %lx (+%lx)...",
                       callstack->function_names[i], callstack->pc[i].virt,
                       callstack->pc[i].relative);
                res = cudbgAPI->resumeWarpsUntilPC(wi->dev, wi->sm,
                                                   (0x1 << wi->warp),
                                                   callstack->pc[i].virt + 0x8);
                if (res != CUDBG_SUCCESS) {
                    LOGE(LOG_ERROR, "cuda error: %s", cudbgGetErrorString(res));
                    goto cleanup;
                }

                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    LOGE(LOG_INFO, "waiting for sm to break...");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                stepup = true;
            }

            if (!cricket_elf_pc_info(callstack->function_names[i],
                                     callstack->pc[i].relative, &rel_ssy,
                                     &rel_pbk)) {
                LOGE(LOG_ERROR, "pc info failed");
                goto cleanup;
            }
            LOGE(LOG_DEBUG, "ssy: %lx > relative: %lx ?", rel_ssy,
                   callstack->pc[i].relative);
            if (rel_ssy > callstack->pc[i].relative) {
                LOGE(LOG_INFO, "joining to pc %lx", callstack->pc[i].virt -
                                                  callstack->pc[i].relative +
                                                  rel_ssy);

                res = cudbgAPI->resumeWarpsUntilPC(
                    wi->dev, wi->sm, (0x1 << wi->warp),
                    callstack->pc[i].virt - callstack->pc[i].relative +
                        rel_ssy);
                if (res != CUDBG_SUCCESS) {
                    LOGE(LOG_DEBUG, "cuda error: %s", cudbgGetErrorString(res));
                    goto cleanup;
                }
                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    LOGE(LOG_DEBUG, "waiting for sm to break...");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                LOGE(LOG_INFO, "joined divergent threads");
                joined = true;
            }
            if (rel_pbk > callstack->pc[i].relative) {
                LOGE(LOG_INFO, "breaking to pc %lx", callstack->pc[i].virt -
                                                   callstack->pc[i].relative +
                                                   rel_pbk);

                res = cudbgAPI->resumeWarpsUntilPC(
                    wi->dev, wi->sm, (0x1 << wi->warp),
                    callstack->pc[i].virt - callstack->pc[i].relative +
                        rel_pbk);
                if (res != CUDBG_SUCCESS) {
                    LOGE(LOG_ERROR, "Cuda Error: \"%s\"", cudbgGetErrorString(res));
                    goto cleanup;
                }
                if (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                    LOGE(LOG_INFO, "waiting for sm to break...");
                    while (!cricket_cr_sm_broken(cudbgAPI, wi->dev, wi->sm)) {
                        usleep(500);
                    }
                }
                LOGE(LOG_INFO, "break'ed divergent threads");
                joined = true;
            }

            if (joined || stepup) {
                cricket_cr_free_callstack(callstack);

                if (!cricket_cr_callstack(cudbgAPI, wi, CRICKET_CR_NOLANE,
                                          callstack)) {
                    LOGE(LOG_ERROR, "failed to get callstack");
                    goto cleanup;
                }
                LOGE(LOG_DEBUG, "new callstack size %d", callstack->callstack_size);
                if (callstack->callstack_size != callstack_bku - i) {
                    LOGE(LOG_ERROR, "new callstack has wrong size");
                    goto cleanup;
                }
                if (joined &&
                    callstack->valid_lanes != callstack->active_lanes) {
                    LOGE(LOG_ERROR, "joning failed, threads still "
                                    "divergent @ rel PC %lx",
                            callstack->pc[0].relative);
                    goto cleanup;
                }
            }
            break;
        }
    }

    if (stepup) {
        LOG(LOG_INFO, "warp was stepped up");
    } else {
        LOG(LOG_INFO, "no up stepping required");
    }
    if (joined) {
        LOG(LOG_INFO, "threads were joined");
    } else {
        LOG(LOG_INFO, "no joining required");
    }
    LOG(LOG_INFO, "threads are now checkpointable");

    ret = true;
cleanup:
    return ret;
}


/* stores stack, registers and PC */
#define CRICKET_PROFILE 1
bool cricket_cr_ckp_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir)
{
    CUDBGResult res;
    size_t register_size;
    void *mem = NULL;
    char *suffix;
    bool ret = false;
#ifdef CRICKET_PROFILE
    struct timeval b, c, d, e, g;
    double ct, dt, et, gt, comt;
#endif

    if (!cricket_cr_gen_suffix(&suffix, wi, lane)) {
        return false;
    }

    if ((mem = malloc(wi->stack_size)) == NULL) {
        LOGE(LOG_ERROR, "error during memory allocation of size %d",
                wi->stack_size);
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&b, NULL);
#endif

    if (!cricket_stack_get_mem(cudbgAPI, wi->dev, wi->sm, wi->warp, lane, mem,
                               wi->stack_size)) {
        fprintf(stderr, "cricket-cr: error while retrieving stack memory\n");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("stack-mem: ");
        for (int i = 0; i != wi->stack_size; ++i) {
            printf("%02x ", ((uint8_t *)mem)[i]);
        }
        printf("\n");
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&c, NULL);
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_STACK, suffix, mem,
                                wi->stack_size)) {
        LOGE(LOG_ERROR, "error writing stack memory");
        goto cleanup;
    }
    register_size = cricket_register_size(wi->dev_prop);
    if ((mem = realloc(mem, register_size)) == NULL) {
        LOGE(LOG_ERROR, "error during memory allocation of size %lu", register_size);
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&d, NULL);
#endif

    if (!cricket_register_ckp(cudbgAPI, wi->dev, wi->sm, wi->warp, lane, mem,
                              wi->dev_prop)) {
        LOGE(LOG_ERROR, "error retrieving register data");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("register-data: ");
        for (int i = 0; i != register_size / sizeof(uint32_t); ++i) {
            printf("%08x ", ((uint32_t *)mem)[i]);
        }
        printf("\n");
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&e, NULL);
#endif

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_REGISTERS, suffix, mem,
                                register_size)) {
        LOGE(LOG_ERROR, "error writing registers");
        goto cleanup;
    }

    if ((mem = realloc(mem, sizeof(uint64_t))) == NULL) {
        LOGE(LOG_ERROR, "error during memory allocation of size %lu",
                sizeof(uint64_t));
        goto cleanup;
    }
#ifdef CRICKET_PROFILE
    gettimeofday(&g, NULL);
    ct = ((double)((c.tv_sec * 1000000 + c.tv_usec) -
                          (b.tv_sec * 1000000 + b.tv_usec))) /
                1000000.;
    dt = ((double)((d.tv_sec * 1000000 + d.tv_usec) -
                          (c.tv_sec * 1000000 + c.tv_usec))) /
                1000000.;
    et = ((double)((e.tv_sec * 1000000 + e.tv_usec) -
                          (d.tv_sec * 1000000 + d.tv_usec))) /
                1000000.;
    gt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                          (e.tv_sec * 1000000 + e.tv_usec))) /
                1000000.;
    comt = ((double)((g.tv_sec * 1000000 + g.tv_usec) -
                            (b.tv_sec * 1000000 + b.tv_usec))) /
                  1000000.;
    LOGE(LOG_DEBUG, "lane time:\n\t\tPROFILE getstack: %f s\n\t\tPROFILE storestack: %f "
           "s\n\t\tPROFILE getreg: %f s\n\t\tPROFILE storereg: %f "
           "s\n\t\tPROFILE lanecomplete: %f s",
           ct, dt, et, gt, comt);
#endif

    ret = true;
cleanup:
    free(suffix);
    free(mem);
    return ret;
}

bool cricket_cr_rst_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *param_mem = NULL;
    void *param_data = NULL;
    size_t heapsize;
    char heap_suffix[8];
    bool ret = false;
    /* Parameters are the same for all warps so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if ((param_mem = (uint8_t*)malloc(elf_info->param_size)) == NULL)
        return false;

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_PARAM, NULL, param_mem,
                               elf_info->param_size)) {
        LOGE(LOG_ERROR, "error reading param");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("param-mem: ");
        for (int i = 0; i != elf_info->param_size; ++i) {
            printf("%02x ", param_mem[i]);
        }
        printf("\n");
    }

    res = cudbgAPI->writeParamMemory(dev, sm, warp,
                                     (uint64_t)elf_info->param_addr, param_mem,
                                     (uint32_t)elf_info->param_size);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }
    for (int i = 0; i != elf_info->param_num; ++i) {
        if (elf_info->params[i].size != 8)
            continue;

        sprintf(heap_suffix, "-P%u", elf_info->params[i].index);

        if (!cricket_file_exists(ckp_dir, CRICKET_DT_HEAP, heap_suffix)) {
            LOGE(LOG_WARNING, "no checkpoint file for parameter %u", i);
            continue;
        }
        param_data = NULL;
        if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                        &param_data, 0, &heapsize)) {
            LOGE(LOG_ERROR, "cricket error while reading heap param data");
            goto cleanup;
        }
        IFLOG(LOG_DBG(3)) {
            printf("heap param %u: %llx (%u)\n", i,
                   *(void **)(param_mem + elf_info->params[i].offset), heapsize);
            printf("param-data for param %u: ", i);
            for (int i = 0; i != heapsize; ++i) {
                printf("%02x ", ((uint8_t *)param_data)[i]);
            }
            printf("\n");
        }

        res = cudbgAPI->writeGlobalMemory(
            *(uint64_t *)(param_mem + elf_info->params[i].offset), param_data,
            heapsize);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        free(param_data);
        param_data = NULL;
    }
    ret = true;
cleanup:
    free(param_mem);
    free(param_data);
    return ret;
}

bool cricket_cr_ckp_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *param_mem = NULL;
    void *param_data = NULL;
    size_t heapsize;
    char heap_suffix[8];
    bool ret = false;
    /* Parameters are the same for all warps so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if ((param_mem = (uint8_t*)malloc(elf_info->param_size)) == NULL)
        return false;

    res =
        cudbgAPI->readParamMemory(dev, sm, warp, (uint64_t)elf_info->param_addr,
                                  param_mem, (uint32_t)elf_info->param_size);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("param-mem: ");
        for (int i = 0; i != elf_info->param_size; ++i) {
            printf("%02x ", param_mem[i]);
        }
        printf("\n");
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_PARAM, NULL, param_mem,
                                elf_info->param_size)) {
        LOGE(LOG_ERROR, "error writing param");
        goto cleanup;
    }
    cricket_focus_host(0);
    for (int i = 0; i != elf_info->param_num; ++i) {
        //continue;
        if (elf_info->params[i].size != 8)
            continue;

        if (!cricket_heap_memreg_size(
                 *(void **)(param_mem + elf_info->params[i].offset),
                 &heapsize)) {
            LOGE(LOG_DEBUG, "cricket-heap: param %u is a 64 bit parameter but does not "
                   "point to an allocated region or an error occured",
                   i);
            continue;
        }

        LOGE(LOG_DEBUG, "heap param %u: %llx (%lu)", i,
               *(void **)(param_mem + elf_info->params[i].offset), heapsize);

        if ((param_data = realloc(param_data, heapsize)) == NULL) {
            goto cleanup;
        }

        res = cudbgAPI->readGlobalMemory(
            *(uint64_t *)(param_mem + elf_info->params[i].offset), param_data,
            heapsize);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        IFLOG(LOG_DBG(3)) {
            printf("param-data for param %u: ", i);
            for (int i = 0; i != heapsize; ++i) {
                printf("%02x ", ((uint8_t *)param_data)[i]);
            }
            printf("\n");
        }

        sprintf(heap_suffix, "-P%u", elf_info->params[i].index);
        if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                    param_data, heapsize)) {
            LOGE(LOG_ERROR, "cricket error while writing param heap");
        }
    }
    ret = true;
cleanup:
    free(param_mem);
    free(param_data);
    return ret;
}

bool cricket_cr_ckp_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *shared_mem = NULL;
    char warp_suffix[16];
    bool ret = false;
    if (elf_info->shared_size == 0)
        return true;

    if ((shared_mem = (uint8_t*)malloc(elf_info->shared_size)) == NULL)
        return false;

    sprintf(warp_suffix, "-D%uS%uW%u", dev, sm, warp);

    res = cudbgAPI->readSharedMemory(dev, sm, warp, 0x0LLU, shared_mem,
                                     (uint32_t)elf_info->shared_size);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("shared-mem (%u): ", elf_info->shared_size);
        for (int i = 0; i != elf_info->shared_size; ++i) {
            printf("%02x ", shared_mem[i]);
        }
        printf("\n");
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_SHARED, warp_suffix,
                                shared_mem, elf_info->shared_size)) {
        LOGE(LOG_ERROR, "error writing param");
        goto cleanup;
    }
    ret = true;
cleanup:
    free(shared_mem);
    return ret;
}

bool cricket_cr_rst_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp)
{
    CUDBGResult res;
    uint8_t *shared_mem = NULL;
    char warp_suffix[16];
    bool ret = false;
    if (elf_info->shared_size == 0)
        return true;

    if ((shared_mem = (uint8_t*)malloc(elf_info->shared_size)) == NULL)
        return false;

    sprintf(warp_suffix, "-D%uS%uW%u", dev, sm, warp);

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_SHARED, warp_suffix,
                               shared_mem, elf_info->shared_size)) {
        LOGE(LOG_ERROR, "error reading shared");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("shared-mem (%u): ", elf_info->shared_size);
        for (int i = 0; i != elf_info->shared_size; ++i) {
            printf("%02x ", shared_mem[i]);
        }
        printf("\n");
    }

    res = cudbgAPI->writeSharedMemory(dev, sm, warp, 0x0LLU, shared_mem,
                                      (uint32_t)elf_info->shared_size);
    if (res != CUDBG_SUCCESS) {
        LOGE(LOG_ERROR, "cuda error: %s",
                cudbgGetErrorString(res));
        goto cleanup;
    }

    ret = true;
cleanup:
    free(shared_mem);
    return ret;
}

bool cricket_cr_rst_globals(CUDBGAPI cudbgAPI, const char *ckp_dir)
{
    CUDBGResult res;
    uint8_t *globals_mem = NULL;
    void *globals_data = NULL;
    size_t heapsize;
    char *heap_suffix = NULL;
    bool ret = false;
    size_t i;
    cricket_global_var *globals;
    size_t globals_num;
    size_t globals_mem_size = 0;
    size_t offset = 0;

    /* Globals are the same for all warps and SMs so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if (!cricket_elf_get_global_vars_info(&globals, &globals_num)) {
        LOGE(LOG_ERROR, "cricket-cr: error getting global variable info");
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        globals_mem_size += globals[i].size;
    }

    if ((globals_mem = (uint8_t*)malloc(globals_mem_size)) == NULL) {
        return false;
    }

    if (!cricket_file_read_mem(ckp_dir, CRICKET_DT_GLOBALS, NULL, globals_mem,
                               globals_mem_size)) {
        LOGE(LOG_ERROR, "error while reading globals");
        goto cleanup;
    }

    IFLOG(LOG_DBG(3)) {
        printf("globals-mem: ");
        for (int i = 0; i != globals_mem_size; ++i) {
            printf("%02x ", globals_mem[i]);
        }
        printf("\n");
    }

    for (i = 0; i < globals_num; ++i) {
        res = cudbgAPI->writeGlobalMemory(
            globals[i].address, globals_mem + offset, globals[i].size);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_DEBUG, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        offset += globals[i].size;
    }

    offset = 0;
    for (i = 0; i != globals_num; ++i) {
        if (globals[i].size != 8)
            continue;

        asprintf(&heap_suffix, "-G%s", globals[i].symbol);

        if (!cricket_file_exists(ckp_dir, CRICKET_DT_HEAP, heap_suffix)) {
            LOGE(LOG_ERROR, "no checkpoint file for global variable %s",
                   globals[i].symbol);
            free(heap_suffix);
            heap_suffix = NULL;
            continue;
        }
        globals_data = NULL;
        if (!cricket_file_read_mem_size(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                        &globals_data, 0, &heapsize)) {
            LOGE(LOG_ERROR, "cricket error while writing globals heap");
            free(heap_suffix);
            heap_suffix = NULL;
            goto cleanup;
        }
        free(heap_suffix);
        heap_suffix = NULL;

        IFLOG(LOG_DBG(3)) {
            printf("heap global %u: %llx (%u)\n", i,
                   *(void **)(globals_mem + offset), heapsize);
            printf("globals-data for global variable %s: ", globals[i].symbol);
            for (int i = 0; i != heapsize; ++i) {
                printf("%02x ", ((uint8_t *)globals_data)[i]);
            }
            printf("\n");
        }

        res = cudbgAPI->writeGlobalMemory(*(uint64_t *)(globals_mem + offset),
                                          globals_data, heapsize);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        free(globals_data);
        globals_data = NULL;

        offset += globals[i].size;
    }
    ret = true;
cleanup:
    free(globals_mem);
    free(globals_data);
    return ret;
}

bool cricket_cr_ckp_globals(CUDBGAPI cudbgAPI, const char *ckp_dir)
{
    CUDBGResult res;
    uint8_t *globals_mem = NULL;
    void *globals_data = NULL;
    size_t heapsize;
    char *heap_suffix = NULL;
    bool ret = false;
    size_t i;
    cricket_global_var *globals;
    size_t globals_num;
    size_t globals_mem_size = 0;
    size_t offset = 0;

    /* Globals are the same for all warps and SMs so just use warp 0
     * TODO: use first valid warp, because warp 0 may not be in use (is that
     * possible?)
     */
    if (!cricket_elf_get_global_vars_info(&globals, &globals_num)) {
        LOGE(LOG_ERROR, "error getting global variable info");
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        globals_mem_size += globals[i].size;
    }

    if ((globals_mem = (uint8_t*)malloc(globals_mem_size)) == NULL) {
        return false;
    }

    for (i = 0; i < globals_num; ++i) {
        res = cudbgAPI->readGlobalMemory(globals[i].address,
                                         globals_mem + offset, globals[i].size);
        if (res != CUDBG_SUCCESS) {
            LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }
        offset += globals[i].size;
    }

    IFLOG(LOG_DBG(3)) {
        printf("globals-mem: ");
        for (int i = 0; i != globals_mem_size; ++i) {
            printf("%02x ", globals_mem[i]);
        }
        printf("\n");
    }

    if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_GLOBALS, NULL, globals_mem,
                                globals_mem_size)) {
        LOGE(LOG_ERROR, "error writing globals");
        goto cleanup;
    }

    offset = 0;
    cricket_focus_host(0);
    for (i = 0; i != globals_num; ++i, offset += globals[i].size) {
        //continue;
        if (globals[i].size != 8)
            continue;

        if (!cricket_heap_memreg_size(*(void **)(globals_mem + offset),
                                      &heapsize)) {
            LOGE(LOG_WARNING, "global variable %s has a size of 64 bit but "
                   "does not point to an allocated region or an error "
                   "occured",
                   globals[i].symbol);
            continue;
        }
        LOGE(LOG_DEBUG,"heap param %u: %llx (%lu)", i,
               *(void **)(globals_mem + offset), heapsize);

        if ((globals_data = realloc(globals_data, heapsize)) == NULL) {
            goto cleanup;
        }

        res = cudbgAPI->readGlobalMemory(*(uint64_t *)(globals_mem + offset),
                                         globals_data, heapsize);
        if (res != CUDBG_SUCCESS) {
             LOGE(LOG_ERROR, "cuda error: %s",
                    cudbgGetErrorString(res));
            goto cleanup;
        }

        IFLOG(LOG_DBG(3)) {
            printf("global-data for global variable %s: ", globals[i].symbol);
            for (int i = 0; i != heapsize; ++i) {
                printf("%02x ", ((uint8_t *)globals_data)[i]);
            }
            printf("\n");
        }

        asprintf(&heap_suffix, "-G%s", globals[i].symbol);
        if (!cricket_file_store_mem(ckp_dir, CRICKET_DT_HEAP, heap_suffix,
                                    globals_data, heapsize)) {
            LOGE(LOG_ERROR, "cricket error while writing param heap");
        }
        free(heap_suffix);
        heap_suffix = NULL;
    }
    ret = true;
cleanup:
    free(globals_mem);
    free(globals_data);
    return ret;
}
