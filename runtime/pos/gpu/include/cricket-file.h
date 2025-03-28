#ifndef _CRICKET_FILE_H_
#define _CRICKET_FILE_H_

#include <stddef.h>
#include <stdbool.h>

typedef enum _cricketDataType {
    CRICKET_DT_REGISTERS = 1,
    CRICKET_DT_PC = 2,
    CRICKET_DT_GLOBALS = 3,
    CRICKET_DT_STACK = 4,
    CRICKET_DT_PARAM = 5,
    CRICKET_DT_HEAP = 6,
    CRICKET_DT_SHARED = 7,
    CRICKET_DT_CALLSTACK = 8,
    CRICKET_DT_LAST
} cricketDataType;

const char *cricket_file_dt2str(cricketDataType data_type);
bool cricket_file_store_mem(const char *path, cricketDataType data_type,
                            const char *suffix, void *data, size_t size);
bool cricket_file_read_mem(const char *path, cricketDataType data_type,
                           const char *suffix, void *data, size_t size);
bool cricket_file_read_mem_size(const char *path, cricketDataType data_type,
                                const char *suffix, void **data,
                                size_t alloc_size, size_t *size);
bool cricket_file_exists(const char *path, cricketDataType data_type,
                         const char *suffix);

#endif //_CRICKET_FILE_H_
