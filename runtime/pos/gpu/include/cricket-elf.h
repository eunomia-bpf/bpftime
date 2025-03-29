#ifndef _CRICKET_ELF_H_
#define _CRICKET_ELF_H_

#include "cricket-types.h"
#include "cudadebugger.h"
#include <stddef.h>

#define CRICKET_ELF_NV_INFO_PREFIX ".nv.info"
#define CRICKET_ELF_NV_SHARED_PREFIX ".nv.shared."
#define CRICKET_ELF_NV_TEXT_PREFIX ".nv.text."
#define CRICKET_ELF_TEXT_PREFIX ".text."

#define CRICKET_ELF_FATBIN ".nv_fatbin"

#define EIFMT_SVAL 4
#define EIFMT_HVAL 3
#define EIFMT_NVAL 1

#define EIATTR_PARAM_CBANK 10
#define EIATTR_CBANK_PARAM_SIZE 25
#define EIATTR_KPARAM_INFO 23
#define EIATTR_MAXREG_COUNT 27
#define EIATTR_S2RCTAID_INSTR_OFFSETS 29
#define EIATTR_EXIT_INSTR_OFFSETS 28
#define EIATTR_EXTERNS 15
#define EIATTR_CRS_STACK_SIZE 30
#define EIATTR_MAX_STACK_SIZE 35
#define EIATTR_MIN_STACK_SIZE 18 // maximal size of the stack when calling this kernel
#define EIATTR_FRAME_SIZE 17 // size of stack in this function (without subcall)

#define CRICKET_SASS_NOP 0x50b0000000070f00
#define CRICKET_SASS_JMX(REG) (0xe20000000007000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_BRX(REG) (0xe25000000007000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_BRX_PRED(REG) (0xe25000000000000f | ((REG & 0xff) << 8))
#define CRICKET_SASS_SSY(ADDR)                                                 \
    (0xe290000000000000 | ((ADDR & 0xffffffff) << 20))
#define CRICKET_SASS_PRET(ADDR)                                                \
    (0xe270000000000040 | ((ADDR & 0xffffffff) << 20))
#define CRICKET_SASS_SYNC(PRED) (0xf0f800000000000f | ((PRED & 0xf) << 20))
#define CRICKET_SASS_EXIT 0xe30000000007000f
#define CRICKET_SASS_CONTROL ((0x5L << 42) + (0x5L << 21) + 0x5L)
#define CRICKET_SASS_FCONTROL 0x001ffc00fd4007ef

void cricket_elf_free_info(cricket_elf_info *info);

bool cricket_elf_get_info(const char *function_name, cricket_elf_info *info);

bool cricket_elf_get_sass_info(const char *filename, const char *section_name,
                               uint64_t relative_pc, cricket_sass_info *info);

bool cricket_elf_restore_patch(const char *filename, const char *new_filename,
                               cricket_callstack *callstack);
bool cricket_elf_get_global_vars_info(cricket_global_var **globals,
                                      size_t *globals_size);

bool cricket_elf_pc_info(const char *function_name, uint64_t relative_pc,
                         uint64_t *relative_ssy, uint64_t *relative_pbk);

bool cricket_elf_analyze(const char *filename);
bool cricket_elf_patch_all(const char *filename, const char *new_filename,
                           cricket_jmptable_index **jumptable,
                           size_t *jumptable_len);

bool cricket_elf_build_fun_info(cricket_function_info **function_info,
                                size_t *fi_num);
bool cricket_elf_get_fun_info(cricket_function_info *function_info,
                              size_t fi_num, const char *fun_name,
                              cricket_function_info **the_fi);
void cricket_elf_free_jumptable(cricket_jmptable_index **jmptbl,
                                size_t jmptbl_len);

bool cricket_elf_get_jmptable_index(cricket_jmptable_index *jmptbl,
                                    size_t jmptbl_len, const char *fn,
                                    cricket_jmptable_index **entry);
bool cricket_elf_get_jmptable_addr(cricket_jmptable_entry *entries,
                                   size_t entries_num, uint64_t destination,
                                   uint64_t *address);

bool cricket_elf_print_symtab(bfd *abfd);

bool cricket_elf_extract_attribute(bfd *obfd,
                                   asection *section,
                                   uint8_t attribute, uint16_t size,
                                   char *data,
                                   bool (*filter_func)(void *, void *),
                                   void *filter_data);

bool cricket_elf_extract_multiple_attributes(bfd *obfd,
                                             asection *section,
                                             uint8_t attribute,
                                             uint16_t size, void **data,
                                             size_t *data_size);

bool stack_size_filter(void *data, void *kernel_i);

bool cricket_elf_extract_shared_size(asection *section,
                                     size_t *size);

#endif //_CRICKET_ELF_H_
