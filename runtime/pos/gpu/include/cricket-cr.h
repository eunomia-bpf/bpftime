#ifndef _CRICKET_CR_H_
#define _CRICKET_CR_H_

#include "cricket-types.h"
#include <stddef.h>
#include <stdbool.h>
#include "cudadebugger.h"

#define CRICKET_CR_NOLANE (0xFFFFFFFF)
#define CRICKET_JMX_ADDR_REG 0

bool cricket_cr_read_pc(cricketWarpInfo *wi, uint32_t lane, const char *ckp_dir,
                        cricket_callstack *callstack);
bool cricket_cr_rst_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_num, cricket_callstack *callstack);
bool cricket_cr_ckp_pc(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                       uint32_t lane_param, const char *ckp_dir,
                       cricket_callstack *callstack);
bool cricket_cr_callstack(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                          uint32_t lane_param, cricket_callstack *callstack);

bool cricket_cr_make_checkpointable(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                                    cricket_function_info *fi, size_t fi_num,
                                    cricket_callstack *callstack);

bool cricket_cr_ckp_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir);
bool cricket_cr_rst_lane(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                         const char *ckp_dir);
bool cricket_cr_function_name(uint64_t pc, const char **name);
bool cricket_cr_kernel_name(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                            uint32_t wp, const char **name);
bool cricket_cr_ckp_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp);
bool cricket_cr_rst_params(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp);
bool cricket_cr_ckp_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp);
bool cricket_cr_rst_shared(CUDBGAPI cudbgAPI, const char *ckp_dir,
                           cricket_elf_info *elf_info, uint32_t dev,
                           uint32_t sm, uint32_t warp);
bool cricket_cr_ckp_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi, uint32_t lane,
                        const char *ckp_dir);
bool cricket_cr_join_threads(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm,
                             uint32_t wp);

bool cricket_cr_rst_globals(CUDBGAPI cudbgAPI, const char *ckp_dir);
bool cricket_cr_ckp_globals(CUDBGAPI cudbgAPI, const char *ckp_dir);

bool cricket_cr_rst_ssy(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                        cricket_callstack *callstack, int c_level,
                        cricket_jmptable_entry *ssy, size_t ssy_num,
                        uint64_t *cur_address);
bool cricket_cr_rst_subcall(CUDBGAPI cudbgAPI, cricketWarpInfo *wi,
                            cricket_callstack *callstack, int c_level,
                            cricket_jmptable_entry *cal, size_t cal_num,
                            uint64_t *cur_address);
bool cricket_cr_sm_broken(CUDBGAPI cudbgAPI, uint32_t dev, uint32_t sm);

void cricket_cr_free_callstack(cricket_callstack *callstack);

#endif //_CRICKET_CR_H_
