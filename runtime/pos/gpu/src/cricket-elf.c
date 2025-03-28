// #include "common-defs.h"
#include "errors.h"
#include "cuda-tdep.h"
#include "objfiles.h"
#include <bfd.h>
#include "libbfd.h"
#include "elf-bfd.h"
#include "stdio.h"
#include "cricket-stack.h"
#include "cuda-elf-image.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/sendfile.h>

#include "cricket-elf.h"
#include "log.h"


typedef struct
{
    uint8_t format;
    uint8_t attr;
    uint16_t size;
} cuda_nv_info_t;

typedef union
{
    char *c;
    cuda_nv_info_t *info;
} cuda_magic_cast_t;

bool cricket_file_cpy(const char *source, const char *destination)
{
    int input, output;
    if ((input = open(source, O_RDONLY)) == -1) {
        LOGE(LOG_ERROR, "failed to open source file %s", source);
        return false;
    }
    if ((output = open(destination, O_WRONLY|O_CREAT|O_TRUNC, 0770)) == -1) {
        LOGE(LOG_WARNING, "create file %s resulted in : \"%s\". Deleting existing file.", destination, strerror(errno));
        if (unlink(destination) != 0) {
            LOGE(LOG_ERROR, "could not delete file %s: \"%s\"", destination, strerror(errno));
            close(input);
            return false;
        } else if ((output = open(destination, O_WRONLY|O_CREAT|O_TRUNC, 0770)) == -1) {
            LOGE(LOG_ERROR, "creating destionation file %s failed again: \"%s\"", destination, strerror(errno));
            return false;
        }
    }

    // sendfile will work with non-socket output (i.e. regular file) on Linux
    // 2.6.33+
    off_t bytesCopied = 0;
    struct stat fileinfo = { 0 };
    fstat(input, &fileinfo);
    int result = sendfile(output, input, &bytesCopied, fileinfo.st_size);

    close(input);
    close(output);

    return result != -1;
}

static void cricket_elf_get_symbols(struct objfile *objfile)
{
    struct minimal_symbol *msymbol;
    int i = 0;
    LOGE(LOG_DEBUG, "msym num: %d, cuda_objfile:%d", objfile->per_bfd->minimal_symbol_count,
           objfile->cuda_objfile);
    for (auto msymbol = objfile->msymbols().begin(); msymbol != objfile->msymbols().end(); ++msymbol) {
        if (!*msymbol) {
            i++;
            continue;
        }
        LOGE(LOG_DEBUG, "%d: name: %s, section:%u, size: %lu, type: %u", i++,
               (*msymbol)->mginfo.name, (*msymbol)->mginfo.section, (*msymbol)->size,
               MSYMBOL_TYPE(*msymbol));
        if ((*msymbol)->mginfo.name[0] != '.') {
            LOGE(LOG_DEBUG, "%p\n", MSYMBOL_VALUE_ADDRESS(objfile, *msymbol));

        }
    }
}

#define CRICKET_ELF_CONST_SUFFIX "_const"
bool cricket_elf_get_global_vars_info(cricket_global_var **globals,
                                      size_t *globals_size)
{
    struct objfile *objfile;
    struct minimal_symbol *msymbol;
    cricket_global_var *arr;
    int i = 0;
    const size_t const_suffix_len = strlen(CRICKET_ELF_CONST_SUFFIX);
    auto objfile_adapter = current_program_space->objfiles();

    if (globals == NULL || globals_size == NULL) {
        LOGE(LOG_ERROR, "at least one parameter is a nullptr");
        return false;
    }

    //Count global vars
    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile) {
        if (!*objfile || !(*objfile)->obfd ||
            !(*objfile)->obfd->tdata.elf_obj_data ||
            !(*objfile)->cuda_objfile)
            continue;

        for (auto msymbol = (*objfile)->msymbols().begin();
             msymbol != (*objfile)->msymbols().end(); ++msymbol) {
            if (!*msymbol) {
                continue;
            }
            if (MSYMBOL_TYPE(*msymbol) == mst_data ||
                MSYMBOL_TYPE(*msymbol) == mst_bss ||
                MSYMBOL_TYPE(*msymbol) == mst_abs) {
                i++;
            }
        }
    }

    if ((arr = (cricket_global_var*)malloc(i * sizeof(cricket_global_var))) == NULL) {
        return false;
    }
    *globals_size = i;
    *globals = arr;
    i = 0;

    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile)
    {
        if (!*objfile || !(*objfile)->obfd ||
            !(*objfile)->obfd->tdata.elf_obj_data ||
            !(*objfile)->cuda_objfile)
            continue;

        for (auto msymbol = (*objfile)->msymbols().begin();
             msymbol != (*objfile)->msymbols().end(); ++msymbol) {
            if (!*msymbol) {
                continue;
            }
            if (MSYMBOL_TYPE(*msymbol) == mst_data ||
                MSYMBOL_TYPE(*msymbol) == mst_bss ||
                MSYMBOL_TYPE(*msymbol) == mst_abs) {
                if (strncmp((*msymbol)->mginfo.name + strlen((*msymbol)->mginfo.name) -
                                const_suffix_len,
                            CRICKET_ELF_CONST_SUFFIX, const_suffix_len) == 0) {
                    continue;
                }
                arr[i].symbol = (*msymbol)->mginfo.name;
                arr[i].address = MSYMBOL_VALUE_ADDRESS(*objfile, *msymbol);
                arr[i].size = MSYMBOL_SIZE(*msymbol);
                i++;
            }
        }
    }
    *globals_size = i;

    IFLOG(LOG_DEBUG) {
        LOGE(LOG_DEBUG, "globals list:");
        for (i = 0; i < *globals_size; ++i) {
            printf("\t(%d) %s: %lx (%lx bytes)\n", i, arr[i].symbol, arr[i].address,
                   arr[i].size);
        }
    }
    return true;
}

static bool cricket_elf_get_symindex(bfd *obfd, const char *name,
                                     uint32_t *index)
{
    size_t storage_needed;
    asymbol **symbol_table;
    size_t number_of_symbols;
    uint32_t i;

    storage_needed = bfd_get_symtab_upper_bound(obfd);

    if (storage_needed <= 0) {
        LOGE(LOG_ERROR, "cirekt-stack (%d): error while getting symtab",
                __LINE__);
        return false;
    }
    symbol_table = (asymbol **)malloc(storage_needed);
    number_of_symbols = bfd_canonicalize_symtab(obfd, symbol_table);

    for (i = 0; i < number_of_symbols; i++) {
        if (strcmp(symbol_table[i]->name, name) == 0) {
            *index = i;
            free(symbol_table);
            return true;
        }
    }
    free(symbol_table);
    return false;
}

bool stack_size_filter(void *data, void *kernel_i)
{
    return *((uint32_t *)data) == *((uint32_t *)kernel_i) + 1;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
bool cricket_elf_extract_multiple_attributes(bfd *obfd,
                                             asection *section,
                                             uint8_t attribute,
                                             uint16_t size, void **data,
                                             size_t *data_size)
{
    cuda_nv_info_t *info;
    bool res = false;
    cuda_magic_cast_t trav, range;
    *data_size = 0;
    *data = NULL;

    char *contents = (char*)malloc(section->size);
    if (contents == NULL) {
        goto cleanup;
    }
    if (!bfd_get_section_contents(obfd, section, contents, 0, section->size)) {
        LOGE(LOG_ERROR, "error during bfd_get_section_contents");
        goto cleanup;
    }
    trav.c = (char *)contents;
    range.c = (char *)contents + section->size;
    while (trav.c < range.c) {
        info = trav.info;
        ++trav.info;
        uint32_t d1, d2;
        d1 = *((uint32_t *)trav.c);
        d2 = *((uint32_t *)(trav.c + 4));
        if (info->attr == attribute) {
            if (info->format == EIFMT_SVAL && info->size == size) {
                *data = realloc(*data, (++(*data_size)) * size);
                memcpy(*data + ((*data_size) - 1) * size, trav.c, size);
                res = true;
            } else {
                LOGE(LOG_WARNING,
                        "cricket_elf_extract_multiple_attributes: "
                        "requested attribute found but size does not match or "
                        "attribute is not an EIFMT_SVAL");
                res = false;
                continue;
            }
        }
        if (info->format == EIFMT_SVAL) {
            trav.c += info->size;
        }
    }
cleanup:
    free(contents);
    return res;
}
#pragma GCC diagnostic pop

bool cricket_elf_extract_attribute(bfd *obfd,
                                   asection *section,
                                   uint8_t attribute, uint16_t size,
                                   char *data,
                                   bool (*filter_func)(void *, void *),
                                   void *filter_data)
{
    cuda_nv_info_t *info;
    bool res = false;
    cuda_magic_cast_t trav, range;

    char *contents = (char*)malloc(section->size);
    if (contents == NULL) {
        goto cleanup;
    }
    if (!bfd_get_section_contents(obfd, section, contents, 0, section->size)) {
        LOGE(LOG_ERROR, "error during bfd_get_section_contents");
        goto cleanup;
    }
    trav.c = (char *)contents;
    range.c = (char *)contents + section->size;
    while (trav.c < range.c) {
        info = trav.info;
        ++trav.info;
        uint32_t d1, d2;
        d1 = *((uint32_t *)trav.c);
        d2 = *((uint32_t *)(trav.c + 4));
        if (info->attr == attribute) {
            if (info->format == EIFMT_SVAL && info->size == size) {
                if (!filter_func || filter_func(trav.c, filter_data)) {
                    memcpy(data, trav.c, size);
                    res = true;
                    goto cleanup;
                }
            } else {
                LOGE(LOG_WARNING,
                        " cricket_stack_extract_elf_attribut: warning: "
                        "requested attribute found but size does not match or "
                        "attribute is not an EIFMT_SVAL");
                continue;
            }
        }
        if (info->format == EIFMT_SVAL) {
            trav.c += info->size;
        }
    }
cleanup:
    free(contents);
    return res;
}

bool cricket_elf_extract_shared_size(asection *section,
                                     size_t *size)
{
    if (size == NULL)
        return false;
    *size = section->size;
    return *size > 0;
}

void cricket_elf_free_info(cricket_elf_info *info)
{
    free(info->params);
}

/* mostly the same as cuda_create_tex_map */
bool cricket_elf_get_info(const char *function_name, cricket_elf_info *info)
{
    struct objfile *objfile;
    struct obj_section *osect = NULL;
    Elf_Internal_Shdr *shdr = NULL;
    asection *section = NULL;
    uint32_t prefixlen = strlen(CRICKET_ELF_NV_INFO_PREFIX);
    uint32_t i;
    uint32_t kernel_index;
    char data[8];
    bool ret = false;
    size_t attr_num;
    char *attrs;
    info->shared_size = 0;

    auto objfile_adapter = current_program_space->objfiles();
    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile) {
        if (!*objfile || !(*objfile)->obfd ||
            !(*objfile)->cuda_objfile)
            continue;

        if (!cricket_elf_get_symindex((*objfile)->obfd, function_name,
                                      &kernel_index))
            continue;


        for (section = (*objfile)->obfd->sections; section != NULL;
             section = section->next) {
            if (strncmp(section->name, CRICKET_ELF_NV_INFO_PREFIX,
                        prefixlen + 1) == 0) {
                if (!cricket_elf_extract_attribute(
                         (*objfile)->obfd, section, EIATTR_MIN_STACK_SIZE, 8, data,
                         stack_size_filter, &kernel_index)) {
                    LOGE(LOG_ERROR,
                            "found .nv.info section but could not find "
                            "stack size for kernel %s",
                            function_name);
                }
                info->stack_size = *(uint32_t *)(data + 4);
            } else if (strncmp(section->name, CRICKET_ELF_NV_INFO_PREFIX,
                               prefixlen) == 0 &&
                       strncmp(section->name + prefixlen + 1, function_name,
                               strlen(function_name)) == 0) {
                if (!cricket_elf_extract_multiple_attributes(
                         (*objfile)->obfd, section, EIATTR_KPARAM_INFO, 12, (void **)&attrs,
                         &attr_num)) {
                    LOGE(LOG_ERROR, "found .nv.info.%s section but "
                                    "could not find "
                                    "any EIATTR_KPARAM_INFO attributes",
                            function_name);
                }
                info->params = (cricket_param_info*)malloc(attr_num * sizeof(cricket_param_info));
                info->param_num = attr_num;
                for (int i = 0; i != attr_num; ++i) {
                    info->params[i].index = *(uint16_t *)(attrs + 4 + i * 12);
                    info->params[i].offset = *(uint16_t *)(attrs + 6 + i * 12);
                    info->params[i].size =
                        *(uint8_t *)(attrs + 10 + i * 12) >> 2;
                }
                free(attrs);
                if (!cricket_elf_extract_attribute((*objfile)->obfd, section,
                                                   EIATTR_PARAM_CBANK, 8, data,
                                                   NULL, NULL)) {
                    LOGE(LOG_ERROR, "found .nv.info.%s section but "
                                    "could not find "
                                    "EIATTR_PARAM_CBANK attribute",
                            function_name);
                }
                info->param_size = *(uint16_t *)(data + 6);
                info->param_addr = *(uint16_t *)(data + 4);
            } else if (strncmp(section->name, CRICKET_ELF_NV_SHARED_PREFIX,
                               strlen(CRICKET_ELF_NV_SHARED_PREFIX) - 1) == 0) {
                if (!cricket_elf_extract_shared_size(section,
                                                     &info->shared_size)) {
                    LOGE(LOG_ERROR, "error while reading shared memory size");
                }
            }
        }
    }
    return false;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
static void cricket_elf_print_mem(void *offset, size_t size)
{
    int c = 0;
    for (uint32_t *p = (uint32_t *)offset; (void *)p < (offset + size); ++p) {
        printf("%08x ", *p);
        if (c++ % 4 == 3)
            printf("\n");
    }
    printf("\n");
}
#pragma GCC diagnostic pop

bool cricket_elf_print_symtab(bfd *abfd)
{
    size_t symtab_size, symtab_length;
    asymbol **symtab;

    if ((symtab_size = bfd_get_symtab_upper_bound(abfd)) == -1) {
        LOGE(LOG_ERROR, "cricket-elf: bfd_get_symtab_upper_bound failed");
        return false;
    }

    LOGE(LOG_DEBUG, "symtab size: %lu", symtab_size);

    if ((symtab = (asymbol **)malloc(symtab_size)) == NULL) {
        LOGE(LOG_ERROR, "cricket-elf: malloc symtab failed");
        return false;
    }

    if ((symtab_length = bfd_canonicalize_symtab(abfd, symtab)) == 0) {
        LOGE(LOG_DEBUG, "symtab empty...");
    } else {
        LOGE(LOG_DEBUG, "%lu symtab entries", symtab_length);
    }

    IFLOG(LOG_DBG(1)) {
        LOG(LOG_DBG(1), "symtab:");
        for (int i = 0; i < symtab_length; ++i) {
            printf("\t%d: %s: %lx\n", i, bfd_asymbol_name(symtab[i]),
                   bfd_asymbol_value(symtab[i]));
        }
    }
    free(symtab);
    return true;
}

#define CRICKET_SASS_BPT (0xe3a00000001000c0L)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
static bool cricket_elf_find_bpt(void *function_base, uint64_t relative_pc,
                                 uint64_t *relative_bpt, uint32_t *number)
{
    bool ret = false;
    uint64_t offset;
    uint64_t cur_instr;
    uint64_t rel_bpt = 0;
    uint32_t num = 0;

    if (function_base == NULL) {
        LOGE(LOG_ERROR, "function_base is NULL");
        return false;
    }

    for (offset = 0L; offset < relative_pc; offset += 0x8) {
        cur_instr = *((uint64_t *)(function_base + offset));
        if (cur_instr == CRICKET_SASS_BPT) {
            rel_bpt = offset;
            if (relative_bpt != NULL) {
                *relative_bpt = rel_bpt;
            }
            while (cur_instr == CRICKET_SASS_BPT) {
                num += 1;
                if ((offset + 0x8) % (4 * 8) == 0)
                    offset += 0x10;
                else
                    offset += 0x8;
                cur_instr = *((uint64_t *)(function_base + offset));
            }

            if (number != NULL)
                *number = num;

            return true;
        }
    }
    if (relative_bpt != NULL) {
        *relative_bpt = 0; // BPT can never occur at relative PC 0 as this is
                           // always
                           // a control instruction
    }
    if (number != NULL) {
        *number = 0;
    }
    return true;
}

#define CRICKET_SASS_PBK_PREFIX 0xe2a00000
static bool cricket_elf_find_pbk(void *function_base, uint64_t relative_pc,
                                 uint64_t *relative_pbk)
{
    bool ret = false;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_pbk_pc;
    uint32_t pbk_pc = 0;

    if (function_base == NULL) {
        LOGE(LOG_ERROR, "function_base is NULL");
        return false;
    }

    for (offset = 0L; offset <= relative_pc; offset += 0x8) {
        cur_instr = *((uint64_t *)(function_base + relative_pc - offset));
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_SASS_PBK_PREFIX) {
            rel_pbk_pc = ((cur_instr >> 20) & 0x000ffffffffL);
            // printf ("rel_syn_pc: %lx\n", rel_syn_pc);
            pbk_pc = (relative_pc - offset) + rel_pbk_pc + 0x8;
            if (relative_pbk != NULL) {
                *relative_pbk = pbk_pc;
            }
            return true;
        }
    }
    if (relative_pbk != NULL) {
        *relative_pbk = 0; // SSY can never occur at relative PC 0 as this is
                           // always
                           // a config instruction
    }
    return true;
}

#define CRICKET_SASS_SSY_PREFIX 0xe2900000
static bool cricket_elf_find_ssy(void *function_base, uint64_t relative_pc,
                                 uint64_t *relative_ssy)
{
    bool ret = false;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_syn_pc;
    uint32_t syn_pc = 0;

    if (function_base == NULL) {
        LOGE(LOG_ERROR, "function_base is NULL");
        return false;
    }

    for (offset = 0L; offset <= relative_pc; offset += 0x8) {
        cur_instr = *((uint64_t *)(function_base + relative_pc - offset));
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_SASS_SSY_PREFIX) {
            rel_syn_pc = ((cur_instr >> 20) & 0x000ffffffffL);
            // printf ("rel_syn_pc: %lx\n", rel_syn_pc);
            syn_pc = (relative_pc - offset) + rel_syn_pc + 0x8;
            if (relative_ssy != NULL) {
                *relative_ssy = syn_pc;
            }
            return true;
        }
    }
    if (relative_ssy != NULL) {
        *relative_ssy = 0; // SSY can never occur at relative PC 0 as this is
                           // always
                           // a config instruction
    }
    return true;
}

static bool cricket_elf_count_ssy(void *function_base, size_t function_size,
                                  size_t *ssy_num,
                                  cricket_jmptable_entry *ssy_targets,
                                  size_t ssy_targets_size)
{
    bool ret = false;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_syn_pc;
    uint32_t cur_ssy_num = 0;

    if (function_base == NULL) {
        LOGE(LOG_ERROR, "function_base is NULL");
        return false;
    }

    for (offset = 0L; offset < function_size; offset += 0x8) {
        cur_instr = *((uint64_t *)(function_base + offset));
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_SASS_SSY_PREFIX) {
            if (ssy_targets_size > cur_ssy_num && ssy_targets != NULL) {
                rel_syn_pc = ((cur_instr >> 20) & 0x000ffffffffL);
                ssy_targets[cur_ssy_num].destination =
                    offset + rel_syn_pc + 0x8;
                // printf ("found SSY @ %lx, targets %lx\n", offset,
                // ssy_targets[cur_ssy_num].destiantion);
            }
            cur_ssy_num++;
        }
    }
    if (ssy_num != NULL) {
        *ssy_num = cur_ssy_num;
    }
    return true;
}

#define CRICKET_SASS_JCAL_PREFIX 0xe2200000
#define CRICKET_SASS_PRET_PREFIX 0xe2700000
static bool cricket_elf_count_cal(void *function_base, size_t function_size,
                                  size_t *num, cricket_jmptable_entry *targets,
                                  size_t targets_size)
{
    bool ret = false;
    uint64_t offset;
    uint64_t cur_instr;
    uint32_t rel_target;
    uint32_t cur_num = 0;

    if (function_base == NULL) {
        LOGE(LOG_ERROR, "function_base is NULL");
        return false;
    }

    for (offset = 0L; offset < function_size; offset += 0x8) {
        cur_instr = *((uint64_t *)(function_base + offset));
        if (((cur_instr >> 32) & 0xfff00000L) == CRICKET_SASS_JCAL_PREFIX ||
            ((cur_instr >> 32) & 0xfff00000L) == CRICKET_SASS_PRET_PREFIX) {
            if (targets_size > cur_num && targets != NULL) {
                rel_target = ((cur_instr >> 20) & 0x000ffffffffL);
                targets[cur_num].destination = offset + rel_target + 0x8;
                // printf ("found JCAL @ %lx, targets %lx\n", offset,
                // targets[cur_num].destination);
            }
            cur_num++;
        }
    }
    if (num != NULL) {
        *num = cur_num;
    }
    return true;
}
#pragma GCC diagnostic pop

bool cricket_elf_get_fun_info(cricket_function_info *function_info,
                              size_t fi_num, const char *fun_name,
                              cricket_function_info **the_fi)
{
    if (fun_name == NULL || function_info == NULL || the_fi == NULL)
        return false;

    for (size_t i = 0; i < fi_num; ++i) {
        if (strcmp(function_info[i].name, fun_name) == 0) {
            *the_fi = function_info + i;
            return true;
        }
    }
    *the_fi = NULL;
    return true;
}

bool cricket_elf_build_fun_info(cricket_function_info **function_info,
                                size_t *fi_num)
{
    asection *section = NULL;
    void *contents = NULL;
    bool ret = false;
    size_t ssy_num = 0;
    size_t cal_num = 0;
    uint32_t bpt_num = 0;
    uint64_t relative_bpt;
    size_t fun_num = 0;
    cricket_function_info *fun_i = NULL;
    size_t i = 0;
    size_t text_prefixlen = strlen(CRICKET_ELF_TEXT_PREFIX) - 1;

    auto objfile_adapter = current_program_space->objfiles();
    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile) {
        if (!*objfile || !(*objfile)->obfd || !(*objfile)->obfd->tdata.elf_obj_data ||
            !(*objfile)->cuda_objfile)
            continue;

        for (section = (*objfile)->obfd->sections; section != NULL;
             section = section->next) {
            if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX,
                        text_prefixlen) != 0) {
                continue;
            }
            fun_num++;
        }
    }

    if ((fun_i = (cricket_function_info*)malloc(fun_num * sizeof(cricket_function_info))) == NULL) {
        goto cleanup;
    }

    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile) {
        if (!*objfile || !(*objfile)->obfd || !(*objfile)->obfd->tdata.elf_obj_data ||
            !(*objfile)->cuda_objfile)
            continue;

        for (section = (*objfile)->obfd->sections; section != NULL;
             section = section->next) {
            if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX,
                        text_prefixlen) != 0) {
                continue;
            }

            if ((contents = malloc(section->size)) == NULL) {
                fprintf(stderr, "cricket-elf: malloc failed\n");
                goto cleanup;
            }

            if (!bfd_get_section_contents((*objfile)->obfd, section, contents, 0,
                                          section->size)) {
                fprintf(stderr, "cricket-elf: getting section failed\n");
                goto cleanup;
            }

            // printf("name: %s, index: %d, size %lx, pos:%p\n", section->name,
            // section->index, section->size, (void*)section->filepos);

            bpt_num = 0;
            ssy_num = 0;
            cal_num = 0;
            fun_i[i].name = section->name + text_prefixlen + 1;

            if (!cricket_elf_find_bpt(contents, section->size, &relative_bpt,
                                      &bpt_num)) {
                fprintf(stderr, "cricket-elf: finding bpt instructions "
                                "failed\n");
                goto cleanup;
            }

            // printf("relative_bpt: %lx, bpt_num: %d\n", relative_bpt,
            // bpt_num);
            fun_i[i].room = bpt_num;

            if (bpt_num == 0) {
                // printf("no room in available\n");
                fun_i[i++].has_room = false;
                continue;
            }

            if (!cricket_elf_count_ssy(contents, section->size, &ssy_num, NULL,
                                       0)) {
                fprintf(stderr, "cricket-elf: counting SSYs failed\n");
                goto cleanup;
            }
            if (!cricket_elf_count_cal(contents, section->size, &cal_num, NULL,
                                       0)) {
                fprintf(stderr, "cricket-elf: counting JCALs failed\n");
                goto cleanup;
            }

            if (bpt_num < ssy_num * 2 + cal_num * 2 + 3) {
                // printf("too little room available: required: %u, available:
                // %u\n",
                // ssy_num*2+cal_num*2+2, bpt_num);
                fun_i[i++].has_room = false;
                continue;
            }
            fun_i[i++].has_room = true;
            free(contents);
            contents = NULL;
        }
    }
    if (function_info != NULL)
        *function_info = fun_i;
    if (fi_num != NULL)
        *fi_num = fun_num;
    ret = true;
cleanup:
    if (!ret)
        free(fun_i);
    free(contents);
    return ret;
}

bool cricket_elf_pc_info(const char *function_name, uint64_t relative_pc,
                         uint64_t *relative_ssy, uint64_t *relative_pbk)
{
    char *section_name = NULL;
    asection *section = NULL;
    uint32_t i;
    void *contents = NULL;
    bool ret = false;
    uint64_t ssy, pbk;
    auto objfile_adapter = current_program_space->objfiles();

    if (asprintf(&section_name, ".text.%s", function_name) == -1) {
        LOGE(LOG_ERROR, "cricket-elf: asprintf failed");
        goto cleanup;
    }
    for (auto objfile = objfile_adapter.begin();
         objfile != objfile_adapter.end();
         ++objfile) {
        if (!*objfile || !(*objfile)->obfd || !(*objfile)->obfd->tdata.elf_obj_data ||
            !(*objfile)->cuda_objfile)
            continue;

        if ((section = bfd_get_section_by_name((*objfile)->obfd, section_name)) ==
            NULL) {
            continue;
        }

        if ((contents = malloc(section->size)) == NULL) {
            fprintf(stderr, "cricket-elf: malloc failed\n");
            goto cleanup;
        }

        if (!bfd_get_section_contents((*objfile)->obfd, section, contents, 0,
                                      section->size)) {
            fprintf(stderr, "cricket-elf: getting section failed\n");
            goto cleanup;
        }

        if (section->size < relative_pc) {
            fprintf(stderr, "cricket-elf: section to small for pc\n");
            goto cleanup;
        }
        if (relative_ssy != NULL &&
            !cricket_elf_find_ssy(contents, relative_pc, &ssy)) {
            fprintf(stderr, "cricket-elf: failed to get ssy\n");
            goto cleanup;
        }
        if (relative_pbk != NULL &&
            !cricket_elf_find_pbk(contents, relative_pc, &pbk)) {
            fprintf(stderr, "cricket-elf: failed to get ssy\n");
            goto cleanup;
        }
        if (relative_ssy != NULL)
            *relative_ssy = ssy;
        if (relative_pbk != NULL)
            *relative_pbk = pbk;

        break;
    }
    ret = true;
cleanup:
    free(section_name);
    free(contents);
    return ret;
}

static bool cricket_elf_patch(const char *filename, size_t filepos,
                              void *patch_data, size_t patch_size)
{
    FILE *fd = NULL;
    bool ret = false;
    printf("patch %s @ %zu with data of size %lu\n", filename, filepos,
           patch_size);

    if (filename == NULL || patch_data == NULL) {
        LOGE(LOG_ERROR, "filename or patch_data NULL");
        return false;
    }

    if ((fd = fopen(filename, "r+b")) == NULL) {
        LOGE(LOG_ERROR, "fopen failed");
        return false;
    }

    if (fseek(fd, filepos, SEEK_SET) != 0) {
        LOGE(LOG_ERROR, "cricket-elf: fseek failed");
        goto cleanup;
    }

    if (fwrite(patch_data, patch_size, 1, fd) != 1) {
        LOGE(LOG_ERROR, "cricket-elf: fwrite failed");
        goto cleanup;
    }
    ret = true;
cleanup:
    fclose(fd);
    return ret;
}

size_t cudabfd_size = 0;
int (*orig_cudabfd_stat)(struct bfd *abfd, struct stat* sb);
int cudabfd_stat(struct bfd *bfd, struct stat *sb)
{
    //int ret = orig_cudabfd_stat(bfd, sb);
    sb->st_size = cudabfd_size;
    return 0;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
bool cricket_elf_patch_all(const char *filename, const char *new_filename,
                           cricket_jmptable_index **jumptable,
                           size_t *jumptable_len)
{
    bfd *hostbfd = NULL;
    bfd *cudabfd = NULL;
    asection *section;
    size_t fatbin_pos, fatbin_size;
    FILE *hostbfd_fd = NULL;
    FILE *cudabfd_fd = NULL;
    void *fatbin = NULL;
    bool ret = false;
    uint32_t bpt_num;
    uint64_t relative_bpt;
    size_t ssy_num;
    size_t cal_num;
    uint64_t *data = NULL;
    size_t data_size;
    size_t data_i = 0;
    cricket_jmptable_index *jmptbl;
    size_t jmptbl_i = 0;
    size_t func_num = 0;
    size_t text_prefixlen = strlen(CRICKET_ELF_TEXT_PREFIX) - 1;
    struct bfd_iovec *iovec = NULL;
    const struct bfd_iovec *orig_iovec = NULL;

    if (filename == NULL) {
        LOGE(LOG_ERROR, "filename is NULL");
        return false;
    }

    if (!cricket_file_cpy(filename, new_filename)) {
        LOGE(LOG_ERROR, "cpy failed");
        return false;
    }

    bfd_init();

    if ((hostbfd_fd = fopen(filename, "r+b")) == NULL) {
        LOGE(LOG_ERROR, "fopen failed");
        return false;
    }

    if ((hostbfd = bfd_openstreamr(filename, NULL, hostbfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openr failed on %s");
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        LOGE(LOG_ERROR, "%s has wrong bfd format",
                filename);
        goto cleanup;
    }

    section = bfd_get_section_by_name(hostbfd, CRICKET_ELF_FATBIN);
    if (section == NULL) {
        LOGE(LOG_ERROR, "fatbin section %s not found",
                CRICKET_ELF_FATBIN);
        goto cleanup;
    }

    LOGE(LOG_DEBUG, "name: %s, index: %d, size 0x%lx, pos:%p", section->name,
           section->index, section->size, (void *)section->filepos);
    fatbin_pos = section->filepos + 0x50;
    fatbin_size = section->size - 0x50;

    if ((fatbin = malloc(fatbin_size)) == NULL) {
        goto cleanup;
    }
    if (fseek(hostbfd_fd, fatbin_pos, SEEK_SET) != 0) {
        LOGE(LOG_ERROR, "cricket-elf: fseek failed");
        goto cleanup;
    }
    if (fread(fatbin, fatbin_size, 1, hostbfd_fd) != 1) {
        LOGE(LOG_ERROR, "cricket-elf: fread failed");
        goto cleanup;
    }

    if ((cudabfd_fd = fmemopen(fatbin, fatbin_size, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fmemopen failed");
        goto cleanup;
    }

    if ((cudabfd = bfd_openstreamr(filename, "elf64-little", cudabfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openstreamr failed");
        fclose(cudabfd_fd);
        goto cleanup;
    }

    //We change the iovec of cudabfd so we can report the correct filesize
    //because in-memory files always report a file size of 0, which creates 
    //problems elsewhere
    cudabfd_size = fatbin_size;
    orig_cudabfd_stat = cudabfd->iovec->bstat;
    orig_iovec = cudabfd->iovec;
    iovec = (struct bfd_iovec*)malloc(sizeof(struct bfd_iovec));
    memcpy(iovec, cudabfd->iovec, sizeof(struct bfd_iovec));
    iovec->bstat = cudabfd_stat;
    cudabfd->iovec = iovec;

    if (!bfd_check_format(cudabfd, bfd_object)) {
        LOGE(LOG_ERROR, "wrong bfd format: %s (%d)", bfd_errmsg(bfd_get_error()), bfd_get_error());
       // goto cleanup;
    }

    for (section = cudabfd->sections; section != NULL;
         section = section->next) {
        LOGE(LOG_DBG(1), "section: %s", section->name);
        if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX, text_prefixlen) !=
            0) {
            continue;
        }
        func_num++;
    }

    if ((jmptbl = (cricket_jmptable_index*)calloc(func_num, sizeof(cricket_jmptable_index))) == NULL) {
        goto cleanup;
    }

    for (section = cudabfd->sections; section != NULL;
         section = section->next) {
        if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX, text_prefixlen) !=
            0) {
            continue;
        }

        // printf("name: %s, index: %d, size %lx, pos:%p\n", section->name,
        // section->index, section->size, (void*)section->filepos);

        if (!cricket_elf_find_bpt(fatbin + (size_t)section->filepos,
                                  section->size, &relative_bpt, &bpt_num)) {
            fprintf(stderr, "cricket-elf: finding bpt instructions failed\n");
            goto cleanup;
        }

        LOGE(LOG_DEBUG, "relative_bpt: %lx, bpt_num: %d", relative_bpt, bpt_num);

        if (bpt_num == 0) {
            LOGE(LOG_WARNING, "no room in \"%s\" available", section->name);
            continue;
        }

        if (!cricket_elf_count_ssy(fatbin + (size_t)section->filepos,
                                   section->size, &ssy_num, NULL, 0)) {
            LOGE(LOG_ERROR, "counting SSYs failed");
            goto cleanup;
        }
        if (!cricket_elf_count_cal(fatbin + (size_t)section->filepos,
                                   section->size, &cal_num, NULL, 0)) {
            LOGE(LOG_ERROR, "counting JCALs failed");
            goto cleanup;
        }

        jmptbl[jmptbl_i].function_name =
            strdup(section->name + text_prefixlen + 1);
        jmptbl[jmptbl_i].start_address = relative_bpt - 0x8;
        if (jmptbl[jmptbl_i].start_address % (4 * 8) == 0) {
            jmptbl[jmptbl_i].start_address -= 0x8;
        }
        jmptbl[jmptbl_i].ssy_num = ssy_num;
        jmptbl[jmptbl_i].cal_num = cal_num;

        if (bpt_num < ssy_num * 2 + cal_num * 2 + 3) {
            LOGE(LOG_ERROR, "too little room available: required: %lu, available: %u",
                   ssy_num * 2 + cal_num * 2 + 2, bpt_num);
            continue;
        }

        if ((jmptbl[jmptbl_i].ssy =
                 (cricket_jmptable_entry*)malloc(ssy_num * sizeof(cricket_jmptable_entry))) == NULL) {
            goto cleanup;
        }
        if ((jmptbl[jmptbl_i].cal =
                 (cricket_jmptable_entry*)malloc(cal_num * sizeof(cricket_jmptable_entry))) == NULL) {
            goto cleanup;
        }

        if (!cricket_elf_count_ssy(fatbin + (size_t)section->filepos,
                                   section->size, NULL, jmptbl[jmptbl_i].ssy,
                                   ssy_num)) {
            LOGE(LOG_ERROR, "counting SSYs failed");
            goto cleanup;
        }
        if (!cricket_elf_count_cal(fatbin + (size_t)section->filepos,
                                   section->size, NULL, jmptbl[jmptbl_i].cal,
                                   cal_num)) {
            LOGE(LOG_ERROR, "counting JCALs failed");
            goto cleanup;
        }

        int ctrl_offset = 4 - (relative_bpt % (4 * 8)) / 8;
        data_size =
            sizeof(uint64_t) * ((ssy_num * 2 + cal_num * 2 + 3) +
                                (ssy_num * 2 + cal_num * 2 + 3) / 3 + 2);
        data_i = 0;
        if ((data = (uint64_t*)malloc(data_size)) == NULL) {
            goto cleanup;
        }
        data[data_i++] = CRICKET_SASS_BRX(0);

        for (int i = 0; i < ssy_num; ++i) {
            if (data_i % 4 == ctrl_offset) {
                data[data_i++] = CRICKET_SASS_FCONTROL;
            }
            jmptbl[jmptbl_i].ssy[i].address = relative_bpt + data_i * 0x8;
            data[data_i++] =
                CRICKET_SASS_SSY(jmptbl[jmptbl_i].ssy[i].destination -
                                 jmptbl[jmptbl_i].ssy[i].address - 0x8);
            if (data_i % 4 == ctrl_offset) {
                data[data_i++] = CRICKET_SASS_FCONTROL;
            }
            data[data_i++] = CRICKET_SASS_BRX(0);
        }
        for (int i = 0; i < cal_num; ++i) {
            if (data_i % 4 == ctrl_offset) {
                data[data_i++] = CRICKET_SASS_FCONTROL;
            }
            jmptbl[jmptbl_i].cal[i].address = relative_bpt + data_i * 0x8;
            data[data_i++] =
                CRICKET_SASS_PRET(jmptbl[jmptbl_i].cal[i].destination -
                                  jmptbl[jmptbl_i].cal[i].address);
            if (data_i % 4 == ctrl_offset) {
                data[data_i++] = CRICKET_SASS_FCONTROL;
            }
            data[data_i++] = CRICKET_SASS_JMX(0);
        }

        if (data_i % 4 == ctrl_offset) {
            data[data_i++] = CRICKET_SASS_FCONTROL;
        }
        jmptbl[jmptbl_i].exit_address = relative_bpt + data_i * 0x8;
        data[data_i++] = CRICKET_SASS_EXIT;
        if (data_i % 4 == ctrl_offset) {
            data[data_i++] = CRICKET_SASS_FCONTROL;
        }
        jmptbl[jmptbl_i].sync_address = relative_bpt + data_i * 0x8;
        data[data_i++] = CRICKET_SASS_SYNC(0);
        if (data_i % 4 == ctrl_offset) {
            data[data_i++] = CRICKET_SASS_FCONTROL;
        }
        data[data_i++] = CRICKET_SASS_BRX(0);

        if (data_i * sizeof(uint64_t) > data_size) {
            LOGE(LOG_ERROR, "too much data to write: have %lu, "
                            "need %lu bytes",
                    data_size, data_i * sizeof(uint64_t));
            goto cleanup;
        }

        if (!cricket_elf_patch(new_filename,
                               fatbin_pos + (size_t)section->filepos +
                                   relative_bpt,
                               data, data_i * sizeof(uint64_t))) {
            LOGE(LOG_ERROR, "patching elf unsuccessful");
            goto cleanup;
        }
        free(data);
        data = NULL;
        jmptbl_i++;
    }

    if (jumptable != NULL)
        *jumptable = jmptbl;
    if (jumptable_len != NULL)
        *jumptable_len = jmptbl_i;

    ret = true;
cleanup:
    free(fatbin);
    free(data);
    free(iovec);
    if (cudabfd != NULL) {
        //We need to restore the original iovec because libbfd is caching iovecs
        //and gets trapped in an endless loop if the iovec is unknown
        cudabfd->iovec = orig_iovec;
        bfd_close(cudabfd);
    }
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}

void cricket_elf_free_jumptable(cricket_jmptable_index **jmptbl,
                                size_t jmptbl_len)
{
    for (size_t i = 0; i < jmptbl_len; ++i) {
        free((*jmptbl)[i].function_name);
        free((*jmptbl)[i].ssy);
        free((*jmptbl)[i].cal);
    }
    free(*jmptbl);
}


bool cricket_elf_analyze(const char *filename)
{
    bfd *hostbfd = NULL;
    bfd *cudabfd = NULL;
    asection *section;
    size_t fatbin_pos, fatbin_size;
    FILE *hostbfd_fd = NULL;
    FILE *cudabfd_fd = NULL;
    void *fatbin = NULL;
    bool ret = false;
    uint32_t bpt_num;
    cricket_jmptable_entry *ssy = NULL;
    cricket_jmptable_entry *cal = NULL;
    size_t ssy_num;
    size_t cal_num;
    size_t fixed_num = 4;

    size_t text_prefixlen = strlen(CRICKET_ELF_TEXT_PREFIX) - 1;
    struct bfd_iovec *iovec;

    if (filename == NULL) {
        LOGE(LOG_ERROR, "filename is NULL");
        return false;
    }

    bfd_init();

    if ((hostbfd_fd = fopen(filename, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fopen failed");
        return false;
    }

    if ((hostbfd = bfd_openstreamr(filename, NULL, hostbfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openr failed on %s",
                filename);
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        LOGE(LOG_ERROR, "%s has wrong bfd format",
                filename);
        goto cleanup;
    }

    section = bfd_get_section_by_name(hostbfd, CRICKET_ELF_FATBIN);
    if (section == NULL) {
        LOGE(LOG_ERROR, "fatbin section %s not found",
                CRICKET_ELF_FATBIN);
        goto cleanup;
    }

    IFLOG(LOG_DEBUG) {
        cricket_elf_print_symtab(hostbfd);
        LOG(LOG_DEBUG, "name: %s, index: %d, size %lx, pos:%x", section->name,
               section->index, section->size, (void *)section->filepos);
    }
    fatbin_pos = section->filepos + 0x50;
    fatbin_size = section->size - 0x50;

    if ((fatbin = malloc(fatbin_size)) == NULL) {
        goto cleanup;
    }
    if (fseek(hostbfd_fd, fatbin_pos, SEEK_SET) != 0) {
        LOGE(LOG_ERROR, "fseek failed");
        goto cleanup;
    }
    if (fread(fatbin, fatbin_size, 1, hostbfd_fd) != 1) {
        LOGE(LOG_ERROR, "fread failed");
        goto cleanup;
    }

    if ((cudabfd_fd = fmemopen(fatbin, fatbin_size, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fmemopen failed");
        goto cleanup;
    }

    if ((cudabfd = bfd_openstreamr(filename, NULL, cudabfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openstreamr failed");
        fclose(cudabfd_fd);
        goto cleanup;
    }

    cudabfd_size = fatbin_size;
    orig_cudabfd_stat = cudabfd->iovec->bstat;
    iovec = (struct bfd_iovec*)malloc(sizeof(struct bfd_iovec));
    memcpy(iovec, cudabfd->iovec, sizeof(struct bfd_iovec));
    iovec->bstat = cudabfd_stat;
    cudabfd->iovec = iovec;

    LOGE(LOG_DEBUG, "Symtab size: %d", bfd_get_file_size(cudabfd));

    if (!bfd_check_format(cudabfd, bfd_object)) {
        LOGE(LOG_ERROR, "wrong bfd format: %s (%d)\n", bfd_errmsg(bfd_get_error()), bfd_get_error());
        goto cleanup;
    }

    IFLOG(LOG_DEBUG) {
        cricket_elf_print_mem(fatbin, 0x160);
        cricket_elf_print_symtab(cudabfd);
    }

    for (section = cudabfd->sections; section != NULL;
         section = section->next) {
        if (strncmp(section->name, CRICKET_ELF_TEXT_PREFIX, text_prefixlen) !=
            0) {
            continue;
        }

        LOGE(LOG_DEBUG, "name: %s, index: %d, size %lx, pos:%p", section->name,
               section->index, section->size, (void *)section->filepos);

        if (!cricket_elf_count_ssy(fatbin + (size_t)section->filepos,
                                   section->size, &ssy_num, NULL, 0)) {
            LOGE(LOG_ERROR, "counting SSYs failed");
            goto cleanup;
        }
        if (!cricket_elf_count_cal(fatbin + (size_t)section->filepos,
                                   section->size, &cal_num, NULL, 0)) {
            LOGE(LOG_ERROR, "counting JCALs failed");
            goto cleanup;
        }

        if ((ssy = (cricket_jmptable_entry*)malloc(ssy_num * sizeof(cricket_jmptable_entry))) == NULL) {
            goto cleanup;
        }
        if ((cal = (cricket_jmptable_entry*)malloc(cal_num * sizeof(cricket_jmptable_entry))) == NULL) {
            goto cleanup;
        }

        if (!cricket_elf_count_ssy(fatbin + (size_t)section->filepos,
                                   section->size, NULL, ssy, ssy_num)) {
            LOGE(LOG_ERROR, "counting SSYs failed");
            goto cleanup;
        }
        if (!cricket_elf_count_cal(fatbin + (size_t)section->filepos,
                                   section->size, NULL, cal, cal_num)) {
            LOGE(LOG_ERROR, "counting JCALs failed");
            goto cleanup;
        }

        if (ssy_num == 0 && cal_num == 0) {
            LOG(LOG_INFO, " => function \"%s\" requires %u slot", section->name,
                   2);
        } else {
            LOG(LOG_INFO, " => function \"%s\" requires %lu slots", section->name,
                   fixed_num + ssy_num * 2 + cal_num * 2);
        }
        free(ssy);
        free(cal);
        ssy = NULL;
        cal = NULL;
    }

    ret = true;
cleanup:
    free(fatbin);
    free(ssy);
    free(cal);
    if (cudabfd != NULL)
        bfd_close(cudabfd);
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}

// TODO: include RETs, CONT, BREAK, etc
bool cricket_elf_get_sass_info(const char *filename, const char *section_name,
                               uint64_t relative_pc, cricket_sass_info *info)
{
    bfd *hostbfd = NULL;
    bfd *cudabfd = NULL;
    asection *section;
    size_t fatbin_pos, fatbin_size;
    FILE *hostbfd_fd = NULL;
    FILE *cudabfd_fd = NULL;
    void *fatbin = NULL;
    bool ret = false;
    uint64_t relative_ssy;
    uint64_t relative_bpt;
    uint32_t bpt_num;
    cricket_jmptable_entry *ssy = NULL;
    cricket_jmptable_entry *cal = NULL;

    if (filename == NULL || section_name == NULL) {
        LOGE(LOG_ERROR, "filename or section_name is NULL");
        return false;
    }

    bfd_init();

    if ((hostbfd_fd = fopen(filename, "r+b")) == NULL) {
        LOGE(LOG_ERROR, "fopen failed");
        return false;
    }

    if ((hostbfd = bfd_openstreamr(filename, NULL, hostbfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openr failed on %s",
                filename);
        fclose(hostbfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(hostbfd, bfd_object)) {
        LOGE(LOG_ERROR, "%s has wrong bfd format",
                filename);
        goto cleanup;
    }

    section = bfd_get_section_by_name(hostbfd, CRICKET_ELF_FATBIN);
    if (section == NULL) {
        LOGE(LOG_ERROR, "fatbin section %s not found",
                CRICKET_ELF_FATBIN);
        goto cleanup;
    }

    LOG(LOG_DEBUG, "name: %s, index: %d, size %lx, pos:%p\n", section->name,
           section->index, section->size, (void *)section->filepos);

    fatbin_pos = section->filepos + 0x50;
    fatbin_size = section->size - 0x50;

    if ((fatbin = malloc(fatbin_size)) == NULL) {
        goto cleanup;
    }
    if (fseek(hostbfd_fd, fatbin_pos, SEEK_SET) != 0) {
        LOGE(LOG_ERROR, "fseek failed");
        goto cleanup;
    }
    if (fread(fatbin, fatbin_size, 1, hostbfd_fd) != 1) {
        LOGE(LOG_ERROR, "fread failed");
        goto cleanup;
    }

    if ((cudabfd_fd = fmemopen(fatbin, fatbin_size, "rb")) == NULL) {
        LOGE(LOG_ERROR, "fmemopen failed");
        goto cleanup;
    }

    if ((cudabfd = bfd_openstreamr(filename, NULL, cudabfd_fd)) == NULL) {
        LOGE(LOG_ERROR, "bfd_openstreamr failed");
        fclose(cudabfd_fd);
        goto cleanup;
    }

    if (!bfd_check_format(cudabfd, bfd_object)) {
        LOGE(LOG_ERROR, "wrong bfd format");
        goto cleanup;
    }

    IFLOG(LOG_DBG(1)) {
        cricket_elf_print_symtab(cudabfd);
    }

    if ((section = bfd_get_section_by_name(cudabfd, section_name)) == NULL) {
        LOGE(LOG_ERROR, "error getting section %s",
                section_name);
        goto cleanup;
    }

    LOG(LOG_DEBUG, "name: %s, index: %d, size %lx, pos:%p", section->name,
           section->index, section->size, (void *)section->filepos);

    if (info != NULL) {
        info->fun_offset = fatbin_pos + (size_t)section->filepos;
        info->fun_size = section->size;
    }

    IFLOG(LOG_DBG(1)) {
        cricket_elf_print_mem((fatbin + (size_t)section->filepos), section->size);
    }

    if (!cricket_elf_find_ssy(fatbin + (size_t)section->filepos, relative_pc,
                              &relative_ssy)) {
        LOGE(LOG_ERROR, "error during find_ssy");
        goto cleanup;
    }

    LOGE(LOG_DEBUG, "relative_ssy: %lx", relative_ssy);

    if (info != NULL) {
        info->ssy = relative_ssy;
    }

    if (!cricket_elf_find_bpt(fatbin + (size_t)section->filepos, relative_pc,
                              &relative_bpt, &bpt_num)) {
        LOGE(LOG_ERROR, "finding bpt instructions failed");
        goto cleanup;
    }

    LOG(LOG_DEBUG, "relative_bpt: %lx, bpt_num: %d", relative_bpt, bpt_num);

    if (info != NULL) {
        info->bpt = relative_bpt;
        info->bpt_num = bpt_num;
    }

    /*   size_t ssy_num;
       size_t cal_num;

       if (!cricket_elf_count_ssy(fatbin+(size_t)section->filepos,
       section->size,
       &ssy_num, NULL, 0)) {
           fprintf(stderr, "cricket-elf: counting SSYs failed\n");
           goto cleanup;
       }
       if (!cricket_elf_count_cal(fatbin+(size_t)section->filepos,
       section->size,
       &cal_num, NULL, 0)) {
           fprintf(stderr, "cricket-elf: counting JCALs failed\n");
           goto cleanup;
       }


       if ((ssy = malloc(ssy_num*sizeof(cricket_jmptable_entry))) == NULL) {
           goto cleanup;
       }
       if ((cal = malloc(cal_num*sizeof(cricket_jmptable_entry))) == NULL) {
           goto cleanup;
       }

       if (!cricket_elf_count_ssy(fatbin+(size_t)section->filepos,
       section->size,
       NULL, ssy, ssy_num)) {
           fprintf(stderr, "cricket-elf: counting SSYs failed\n");
           goto cleanup;
       }
       if (!cricket_elf_count_cal(fatbin+(size_t)section->filepos,
       section->size,
       NULL, cal, cal_num)) {
           fprintf(stderr, "cricket-elf: counting JCALs failed\n");
           goto cleanup;
       }*/

    ret = true;
cleanup:
    free(fatbin);
    // free(ssy);
    // free(cal);
    if (cudabfd != NULL)
        bfd_close(cudabfd);
    if (hostbfd != NULL)
        bfd_close(hostbfd);
    return ret;
}
#pragma GCC diagnostic pop

bool cricket_elf_get_jmptable_addr(cricket_jmptable_entry *entries,
                                   size_t entries_num, uint64_t destination,
                                   uint64_t *address)
{
    if (entries == NULL | address == NULL)
        return false;

    for (size_t i = 0; i < entries_num; ++i) {
        if (entries[i].destination == destination) {
            *address = entries[i].address;
            return true;
        }
    }
    *address = 0;
    return true;
}

bool cricket_elf_get_jmptable_index(cricket_jmptable_index *jmptbl,
                                    size_t jmptbl_len, const char *fn,
                                    cricket_jmptable_index **entry)
{
    if (jmptbl == NULL | fn == NULL | entry == NULL)
        return false;

    for (size_t i = 0; i < jmptbl_len; ++i) {
        if (strcmp(fn, jmptbl[i].function_name) == 0) {
            *entry = jmptbl + i;
            return true;
        }
    }
    *entry = NULL;
    return true;
}

bool cricket_elf_restore_patch(const char *filename, const char *new_filename,
                               cricket_callstack *callstack)
{
    char *section_name = NULL;
    bool ret = false;
    cricket_sass_info sinfo;
    cricket_sass_info sinfo_kernel;

    // uint64_t data[8];
    int data_size = 0;
    uint64_t data[] = { 0x001ffc00fd4007ef,
                        // CRICKET_SASS_CONTROL,
                        CRICKET_SASS_NOP,    CRICKET_SASS_NOP,
                        CRICKET_SASS_NOP,    0x001ffc00fd4007ef,
                        CRICKET_SASS_JMX(18) };

    if (!cricket_file_cpy(filename, new_filename)) {
        LOGE(LOG_ERROR, "cpy failed");
        return false;
    }

    if (asprintf(&section_name, ".text.%s",
                 callstack->function_names[callstack->callstack_size - 1]) ==
        -1) {
        LOGE(LOG_ERROR, "asprintf failed");
        return false;
    }
    if (!cricket_elf_get_sass_info(new_filename, section_name,
                                   callstack->pc[callstack->callstack_size - 1]
                                       .relative,
                                   &sinfo_kernel)) {
        LOGE(LOG_ERROR, "cuda function %s not found in elf %s",
                callstack->function_names[0], new_filename);
        goto cleanup;
    }
    free(section_name);
    section_name = NULL;

    if (callstack->callstack_size > 2) {
        LOGE(LOG_ERROR, "cricket-elf: patching for callstack-sizes greater than 2 is not possible");
        return false;
    } else if (callstack->callstack_size == 2) {
        if (asprintf(&section_name, ".text.%s", callstack->function_names[0]) ==
            -1) {
            LOGE(LOG_ERROR, "asprintf failed");
            return false;
        }

        if (!cricket_elf_get_sass_info(new_filename, section_name,
                                       callstack->pc[0].relative, &sinfo)) {
            LOGE(LOG_ERROR, "cuda function %s not found in elf %s",
                    callstack->function_names[0], new_filename);
            goto cleanup;
        }

        LOG(LOG_DEBUG, "fun_offset: %lx, fun_size: %lx, ssy: %lx", sinfo.fun_offset,
               sinfo.fun_size, sinfo.ssy);

        free(section_name);
        section_name = NULL;
        if (sinfo.ssy > callstack->pc[0].relative) {
            LOGE(LOG_ERROR, "cricket-elf: threads in a divergent block cannot be "
                    "restored. something during checkpoint went wrong if "
                    "this occurs here.");
            goto cleanup;
            // data[1] = CRICKET_SASS_SSY(sinfo.ssy-0x8);
            /*if (callstack->active_lanes != callstack->valid_lanes) {
                data[3] = CRICKET_SASS_SYNC(0);
            }*/
        }
        data[2] = CRICKET_SASS_PRET(callstack->pc[1].relative - 0x18);
        data[3] = CRICKET_SASS_JMX(18);
    } else { // callstack == 1
        if (sinfo_kernel.ssy >= callstack->pc[0].relative) {
            data[1] = CRICKET_SASS_SSY(sinfo_kernel.ssy - 0x10);
            if (callstack->active_lanes != callstack->valid_lanes) {
                data[2] = CRICKET_SASS_SYNC(0);
            }
        }
        data[3] = CRICKET_SASS_BRX(18);
    }

    // data[1] = CRICKET_SASS_PRET(0x10);
    // data[2] = CRICKET_SASS_PRET(0x2718);
    // data[3] = CRICKET_SASS_JMX(0x2b3900);

    if (!cricket_elf_patch(new_filename, sinfo_kernel.fun_offset, data,
                           6 * sizeof(uint64_t))) {
        LOGE(LOG_ERROR, "patching elf unsuccessful");
        goto cleanup;
    }

    ret = true;
cleanup:
    free(section_name);
    return ret;
}
