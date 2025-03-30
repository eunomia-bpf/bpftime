#ifndef _CRICKET_TYPES_H_
#define _CRICKET_TYPES_H_

#include <build-gnulib/config.h>
#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>

typedef struct _CricketDeviceProp
{
    uint32_t index;
    char *name;
    char *type;
    char *smType;
    uint32_t numLanes;
    uint32_t numPredicates;
    uint32_t numRegisters;
    uint32_t numSMs;
    uint32_t numWarps;
} CricketDeviceProp;

typedef struct _cricketWarpInfo
{
    uint32_t dev;
    uint32_t sm;
    uint32_t warp;
    const char *kernel_name;
    uint32_t stack_size;
    CricketDeviceProp *dev_prop;
} cricketWarpInfo;

typedef struct _cricketSMInfo
{
    uint32_t dev;
    uint32_t sm;

} cricketSMInfo;

typedef struct _cricket_pc_data
{
    uint64_t relative;
    uint64_t absolute;
    uint64_t virt;
    uint64_t str_offset;
} cricket_pc_data;

typedef struct _cricket_callstack
{
    uint32_t active_lanes;
    uint32_t valid_lanes;
    uint32_t callstack_size;
    cricket_pc_data *pc;
    const char **function_names;
    void *_packed_ptr;
} cricket_callstack;

typedef struct _cricket_param_info
{
    uint16_t index;
    uint16_t ordinal;
    uint16_t offset;
    uint8_t size;
} cricket_param_info;

typedef struct _cricket_elf_info
{
    uint32_t stack_size;
    uint16_t param_addr;
    uint16_t param_size;
    cricket_param_info *params;
    size_t param_num;
    size_t shared_size;
} cricket_elf_info;

// Only required for restore
typedef struct _cricket_sass_info
{
    size_t fun_offset; // Offset in binary where SASS for given function begins
    size_t fun_size;   // Size of .text section
    uint64_t ssy; // PC where the last SSY instruction set a synchronization
                  // point to
    // Relative to function start
    uint64_t bpt;
    uint32_t bpt_num;
} cricket_sass_info;

typedef struct _cricket_global_var
{
    const char *symbol;
    uint64_t address;
    uint64_t size;
} cricket_global_var;

typedef struct _cricket_function_info
{
    const char *name;
    bool has_room;
    size_t room;
} cricket_function_info;

typedef struct _cricket_jmptable_entry
{
    uint64_t address;
    uint64_t destination;
} cricket_jmptable_entry;

typedef struct _cricket_jmptable_index
{
    char *function_name;
    uint64_t start_address;
    size_t ssy_num;
    size_t cal_num;
    cricket_jmptable_entry *ssy;
    cricket_jmptable_entry *cal;
    uint64_t sync_address;
    uint64_t exit_address;
} cricket_jmptable_index;

#endif
