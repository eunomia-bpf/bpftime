/*
 * Copyright 2024 The PhoenixOS Authors. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <map>

#include <libelf.h>
#include <gelf.h>
#include <string.h>

#include "pos/include/common.h"
#include "pos/include/log.h"

#include "pos/include/patcher.h"

#define FATBIN_STRUCT_MAGIC 0x466243b1
#define FATBIN_TEXT_MAGIC   0xBA55ED50

#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_COMPRESS  0x0000000000002000LL

#define EIATTR_PARAM_CBANK              0xa
#define EIATTR_EXTERNS                  0xf
#define EIATTR_FRAME_SIZE               0x11
#define EIATTR_MIN_STACK_SIZE           0x12
#define EIATTR_KPARAM_INFO              0x17
#define EIATTR_CBANK_PARAM_SIZE         0x19
#define EIATTR_MAX_REG_COUNT            0x1b
#define EIATTR_EXIT_INSTR_OFFSETS       0x1c
#define EIATTR_S2RCTAID_INSTR_OFFSETS   0x1d
#define EIATTR_CRS_STACK_SIZE           0x1e
#define EIATTR_SW1850030_WAR            0x2a
#define EIATTR_REGCOUNT                 0x2f
#define EIATTR_SW2393858_WAR            0x30
#define EIATTR_INDIRECT_BRANCH_TARGETS  0x34
#define EIATTR_CUDA_API_VERSION         0x37

#define EIFMT_NVAL                      0x1
#define EIFMT_HVAL                      0x3
#define EIFMT_SVAL                      0x4

/*!
 *  \brief  descriptor of a CUDA function
 */
typedef struct POSCudaFunctionDesp {
    // name of the kernel
    std::string name;
    
    // kernel signature
    std::string signature;

    // number of parameters within this function
    uint32_t nb_params;

    // offset of each parameter
    std::vector<uint32_t> param_offsets;

    // size of each parameter
    std::vector<uint32_t> param_sizes;

    // index of those parameter which is a input pointer (const pointer)
    std::vector<uint32_t> input_pointer_params;

    // index of those parameter which is a inout pointer
    std::vector<uint32_t> inout_pointer_params;

    // index of those parameter which is a output pointer
    std::vector<uint32_t> output_pointer_params;

    // index of those non-pointer parameters that may carry pointer inside their values
    std::vector<uint32_t> suspicious_params;
    bool has_verified_params;

    // confirmed suspicious parameters: index of the parameter -> offset from the base address
    std::vector<std::pair<uint32_t,uint64_t>> confirmed_suspicious_params;

    // cbank parameter size (p.s., what is this?)
    uint64_t cbank_param_size;

    POSCudaFunctionDesp() : nb_params(0), cbank_param_size(0), has_verified_params(false) {}
    ~POSCudaFunctionDesp(){}
} POSCudaFunctionDesp_t;


/*!
 *  \brief  kernel PTX patcher
 */
class POSUtil_CUDA_Kernel_Patcher {
 public:
    static pos_retval_t patch_fatbin_binary(uint8_t *binary_ptr, std::vector<uint8_t>& patched_bianry){
        pos_retval_t retval = POS_SUCCESS;

        POS_CHECK_POINTER(binary_ptr);

        std::unique_ptr<std::vector<uint8_t>> _patched_fatbin = patch_fatbin(binary_ptr);
        if(unlikely(_patched_fatbin == nullptr || _patched_fatbin->size() == 0)){
            POS_WARN("failed to patch fatbin: fatbin(%p)", binary_ptr);
            retval = POS_FAILED_INCORRECT_OUTPUT;
            goto exit;
        }

        patched_bianry = *_patched_fatbin;

    exit:
        return retval;
    }

 private:
};


/*！
 *  \brief  parser of CUDA kernel
 */
class POSUtil_CUDA_Kernel_Parser {
 public:
    /*!
     *  \brief  analyse the behaviour of an kernel based on its prototype, 
     *          the behaviour is represent by its parameter (i.e., whether it's a pointer), and the direction of the
     *          pointer (i.e., whether this pointer is an const pointer)
     *  \param  kernel_str      low-level (mangles) identifiers of the kernel
     *  \param  function_desp   pointer of function descriptor
     *  \example    mangles:    _Z8kernel_1PKfPfS1_S1_i
     *              demangles:  kernel_1(const float *, float *, float *, float *, int)
     *  \note   this function will use binary utilites "cu++filt" to obtain the kernel prototype, and
     *          use clang to parse the semantics of the prototype
     *  \todo   use __cu_demangle under CUDA 12.0
     *  \return POS_SUCCESS for successfully parsing
     *          POS_FAILED for failed parsing
     */
    static pos_retval_t parse_by_prototype(const char *kernel_str, POSCudaFunctionDesp* function_desp){
        pos_retval_t retval = POS_SUCCESS;
        std::string kernel_demangles_name, kernel_prototype;

        POS_CHECK_POINTER(kernel_str);
        POS_CHECK_POINTER(function_desp);

        retval = __preprocess_prototype(kernel_str, kernel_demangles_name);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed parsing kernel prototype: preprocess failed: kernel_str(%s)", kernel_str);
            goto exit;
        }

        retval = __generate_prototype(kernel_demangles_name, kernel_prototype);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed parsing kernel prototype: generate prototype failed: kernel_str(%s)", kernel_str);
            goto exit;
        }

        function_desp->signature = kernel_prototype;

        retval = __parse_prototype(kernel_prototype, function_desp);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "failed parsing kernel prototype: parsing failed: kernel_str(%s), kernel_prototype(%s)",
                kernel_str, kernel_prototype.c_str()
            );
            goto exit;
        }

    exit:
        return retval;
    }
 
 private:
    /*!
     *  \brief  preprocess a raw demangles name
     *  \param  kernel_str              the raw demangles name
     *  \param  kernel_demangles_name   the processed demangles name
     *  \return POS_SUCCESS for successfully processed
     *          POS_INVALID_INPUT for invalid raw demangles name
     */
    static pos_retval_t __preprocess_prototype(const char *kernel_str, std::string& kernel_demangles_name);

    /*!
     *  \brief  generate the origin kernel prototype based on processed demangles name
     *  \param  kernel_demangles_name   the processed demangles name
     *  \param  kernel_prototype        the generated kernel prototype
     *  \return POS_SUCCESS for successfully generation
     *          POS_FAILED for failed generation
     */
    static pos_retval_t __generate_prototype(const std::string& kernel_demangles_name, std::string& kernel_prototype);

    /*!
     *  \brief  parsing the kernel prototype
     *  \param  kernel_prototype        the generated kernel prototype
     *  \param  function_desp           pointer of function descriptor
     *  \return POS_SUCCESS for successfully processed
     *          POS_FAILED for failed processed
     */
    static pos_retval_t __parse_prototype(const std::string& kernel_prototype, POSCudaFunctionDesp* function_desp);
};


/*！
 *  \brief  utilities of CUDA fatbin
 */
class POSUtil_CUDA_Fatbin {
 public:
    /*!
     *  \brief  obtain metadata of CUDA functions from given fatbin
     *  \param  binary_ptr  pointer to the memory area that stores the fatbin
     *                      (note: the content should start from the fatbin ELF header)
     *  \param  binary_size size of the given binary
     *  \param  desps       vector to store the extracted function metadata
     *  \param  cached_deps cached function metadata
     *  \return POS_SUCCESS for successfully extraction
     */
    static pos_retval_t obtain_functions_from_cuda_binary(
        uint8_t* binary_ptr,
        uint64_t binary_size,
        std::vector<POSCudaFunctionDesp*>* desps,
        std::map<std::string, POSCudaFunctionDesp*>& cached_desp_map
    ){
        pos_retval_t retval = POS_SUCCESS;
        POSCudaFunctionDesp* new_desp;
        const uint8_t *input_pos = NULL;
        uint8_t *text_data = NULL;
        size_t text_data_size = 0;
        size_t fatbin_total_size = 0;
        uint32_t nb_text_section=1;
        
        fat_elf_header_t *fatbin_elf_hdr;
        fat_text_header_t *fatbin_text_hdr;

    #define __POS_DUMP_FATBIN 0
    #if __POS_DUMP_FATBIN
        std::ofstream cubin_file("/tmp/ptx.txt", std::ios::out);
        if(unlikely(!cubin_file)){
            POS_ERROR_DETAIL("failed to open /tmp/ptx.txt");
        }
    #endif

        POS_CHECK_POINTER(desps);
        POS_CHECK_POINTER(input_pos = binary_ptr);
        POS_ASSERT(binary_size > 0);

        fatbin_elf_hdr = (fat_elf_header_t*)input_pos;
        retval = POSUtil_CUDA_Fatbin::__verify_fatbin_elf_header(fatbin_elf_hdr);
        if(retval == POS_SUCCESS){
            /*!
             *  \note   case: this is a fatbin that contains multiple cubin
             */
            input_pos += fatbin_elf_hdr->header_size;
            fatbin_total_size = fatbin_elf_hdr->header_size + fatbin_elf_hdr->size;

            do {
                // verify fatbin text header
                fatbin_text_hdr = (fat_text_header_t*)input_pos;
                retval = POSUtil_CUDA_Fatbin::__verify_fatbin_text_header(fatbin_text_hdr);
                if(unlikely(retval != POS_SUCCESS)){
                    goto exit;
                }
                input_pos += fatbin_text_hdr->header_size;

                POS_DEBUG(
                    "parsing %u(th) text section: arch(%u), major(%u), minor(%u)",
                    nb_text_section, fatbin_text_hdr->arch, fatbin_text_hdr->major, fatbin_text_hdr->minor
                );

                // section does not cotain device code (e.g. only PTX)
                if (fatbin_text_hdr->kind == 2) {
                    /*!
                     *  \note   contains SASS code, extract kernel information from ELF format
                     */
                    // this section contains debug info
                    if (fatbin_text_hdr->flags & FATBIN_FLAG_DEBUG){
                        POS_DEBUG("%u(th) fatbin text section contains debug information", nb_text_section);
                    }

                    if (fatbin_text_hdr->flags & FATBIN_FLAG_COMPRESS){
                        // the payload of this section is compressed, need to be decompressed
                        ssize_t input_read;
                        POS_DEBUG(
                            "%u(th) fatbin text section contains compressed device code, decompressing...",
                            nb_text_section
                        );

                        input_read = POSUtil_CUDA_Fatbin::__decompress_single_text_section(
                            input_pos, &text_data, &text_data_size, fatbin_elf_hdr, fatbin_text_hdr
                        );
                        if(unlikely(input_read < 0)){
                            POS_WARN("failed to decompress %u(th) fatbin text section", nb_text_section);
                            retval = POS_FAILED;
                            goto exit;
                        }

                        input_pos += input_read;
                    } else {
                        text_data = (uint8_t*)input_pos;
                        text_data_size = fatbin_text_hdr->size;
                        input_pos += fatbin_text_hdr->size;
                    }

                    retval = POSUtil_CUDA_Fatbin::__extract_kernel_infos(text_data, text_data_size, desps, cached_desp_map);
                    if(unlikely(retval != POS_SUCCESS)){
                        goto exit;
                    }

                    if (fatbin_text_hdr->flags & FATBIN_FLAG_COMPRESS) {
                        free(text_data);
                    }

                    nb_text_section += 1;
                } else {
                    /*!
                     *  \note   contains PTX code only
                     */
                    POS_DEBUG("skip this text section as it doesn't contain any device code");
                    
                #if __POS_DUMP_FATBIN
                    // POS_LOG("addr: %p, len: %lu", input_pos, fatbin_text_hdr->size);
                    cubin_file.write((const char*)(input_pos), fatbin_text_hdr->size);
                    cubin_file.flush();
                    cubin_file.close();
                #endif

                    input_pos += fatbin_text_hdr->size;
                    continue;
                }
            } while(input_pos < (uint8_t*)fatbin_elf_hdr + fatbin_elf_hdr->header_size + fatbin_elf_hdr->size);
        } else {
            /*!
             *  \note   case: this is a single ELF cubin
             */
            retval = POSUtil_CUDA_Fatbin::__extract_kernel_infos(binary_ptr, binary_size, desps, cached_desp_map);
            if(unlikely(retval != POS_SUCCESS)){
                goto exit;
            }
        }

    #undef __POS_DUMP_FATBIN

    exit:
        return retval;
    }


 private:
    /*!
     *  \brief  fatbin ELF header definition
     */
    typedef struct  __attribute__((__packed__)) fat_elf_header {
        uint32_t magic;
        uint16_t version;
        uint16_t header_size;
        uint64_t size;
    } fat_elf_header_t;

    /*!
     *  \brief  fatbin text header definition
     */
    typedef struct  __attribute__((__packed__)) fat_text_header
    {
        uint16_t kind;
        uint16_t unknown1;
        uint32_t header_size;
        uint64_t size;
        uint32_t compressed_size;       // Size of compressed data
        uint32_t unknown2;              // Address size for PTX?
        uint16_t minor;
        uint16_t major;
        uint32_t arch;
        uint32_t obj_name_offset;
        uint32_t obj_name_len;
        uint64_t flags;
        uint64_t zero;                  // Alignment for compression?
        uint64_t decompressed_size;     // Length of compressed data in decompressed representation.
                                        // There is an uncompressed footer so this is generally smaller
                                        // than size.
    } fat_text_header_t;

    struct __attribute__((__packed__)) nv_info_entry{
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t kernel_id;
        uint32_t value;
    };

    struct __attribute__((__packed__)) nv_info_kernel_entry {
        uint8_t format;
        uint8_t attribute;
        uint16_t values_size;
        uint32_t values;
    };

    struct __attribute__((__packed__)) nv_info_kparam_info {
        uint32_t index;
        uint16_t ordinal;
        uint16_t offset;
        uint16_t unknown : 12;
        uint8_t  cbank : 6;
        uint16_t size : 14;
    };

    /*!
     *  \brief  verify the fatbin ELF header
     *  \param  fatbin_elf_hdr  base address of the fatbin ELF header
     *  \return POS_SUCCESS for successfully verifying
     */
    static pos_retval_t __verify_fatbin_elf_header(fat_elf_header_t* fatbin_elf_hdr){
        pos_retval_t retval = POS_SUCCESS;

        POS_CHECK_POINTER(fatbin_elf_hdr);

        if(unlikely(fatbin_elf_hdr->magic != FATBIN_TEXT_MAGIC)){
            POS_DEBUG(
                "invalid magic within the fatbin ELF header: given(%x), expected(%x)",
                fatbin_elf_hdr->magic, FATBIN_TEXT_MAGIC
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }

        if(unlikely(fatbin_elf_hdr->version != 1)){
            POS_WARN(
                "version within the fatbin ELF header is wrong: given(%u), expected(1)",
                fatbin_elf_hdr->version
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }

        if(unlikely(fatbin_elf_hdr->header_size != sizeof(fat_elf_header_t))){
            POS_WARN(
                "header size within the fatbin ELF header is wrong: given(%u), expected(%lu)",
                fatbin_elf_hdr->header_size, sizeof(fat_elf_header_t)
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }

    exit:
        return retval;
    }

    /*!
     *  \brief  verify the fatbin text header
     *  \param  fatbin_text_base    base address of the fatbin text header
     *  \return POS_SUCCESS for successfully verifying
     */
    static pos_retval_t __verify_fatbin_text_header(fat_text_header_t* fatbin_text_base){
        pos_retval_t retval = POS_SUCCESS;
        
        POS_CHECK_POINTER(fatbin_text_base);

        if(fatbin_text_base->obj_name_offset != 0) {
            if (
                ((char*)fatbin_text_base)[fatbin_text_base->obj_name_offset + fatbin_text_base->obj_name_len] != '\0'
            ){
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            } else {
                char *obj_name = (char*)fatbin_text_base + fatbin_text_base->obj_name_offset;
                POS_DEBUG("fatbin object name: %s (len:%#x)", obj_name, fatbin_text_base->obj_name_len);
            }
        }

    exit:
        return retval;
    }

    /*! 
     *  \brief  decompresses a single text section within the fatbin file
     */
    static ssize_t __decompress_single_text_section(
        const uint8_t *input, uint8_t **output, size_t *output_size,
        fat_elf_header_t *eh, fat_text_header_t *th
    ){
        size_t padding;
        size_t input_read = 0;
        size_t output_written = 0;
        size_t decompress_ret = 0;
        const uint8_t zeroes[8] = {0};

        POS_CHECK_POINTER(input); POS_CHECK_POINTER(output);
        POS_CHECK_POINTER(eh); POS_CHECK_POINTER(th);

        // add max padding of 7 bytes
        POS_CHECK_POINTER(*output = (uint8_t*)malloc(th->decompressed_size + 7)); 

        decompress_ret = POSUtil_CUDA_Fatbin::__decompress(input, th->compressed_size, *output, th->decompressed_size);

        if (unlikely(decompress_ret != th->decompressed_size)) {
            POS_WARN(
                "decompression failed: decompressed size is %#zx, but header says %#zx", 
                decompress_ret, th->decompressed_size
            );
            POS_WARN("input pos: %#zx, output pos: %#zx", input - (uint8_t*)eh, *output);
            goto     POSUtil_CUDA_Fatbin___decompress_single_text_section_error;
        }
        input_read += th->compressed_size;
        output_written += th->decompressed_size;

        padding = ((8 - (size_t)(input + input_read)) % 8);
        if (memcmp(input + input_read, zeroes, padding) != 0) {
            POS_WARN("expected %#zx zero bytes, got:", padding);
            goto POSUtil_CUDA_Fatbin___decompress_single_text_section_error;
        }
        input_read += padding;

        padding = ((8 - (size_t)th->decompressed_size) % 8);

        // Because we always allocated enough memory for one more elf_header and this is smaller than
        // the maximal padding of 7, we do not have to reallocate here.
        memset(*output, 0, padding);
        output_written += padding;

        *output_size = output_written;
        return input_read;

    POSUtil_CUDA_Fatbin___decompress_single_text_section_error:
        free(*output);
        *output = NULL;
        return -1;
    }

    /*! 
     *  \brief  decompresses a fatbin file
     *  \param  input       pointer compressed input data
     *  \param  input_size  size of compressed data
     *  \param  output      preallocated memory where decompressed output should be stored
     *  \param  output_size size of output buffer. Should be equal to the size of the decompressed data
     */
    static size_t __decompress(const uint8_t* input, size_t input_size, uint8_t* output, size_t output_size){
        size_t ipos = 0, opos = 0;  
        uint64_t next_nclen;  // length of next non-compressed segment
        uint64_t next_clen;   // length of next compressed segment
        uint64_t back_offset; // negative offset where redudant data is located, relative to current opos

        while (ipos < input_size) {
            next_nclen = (input[ipos] & 0xf0) >> 4;
            next_clen = 4 + (input[ipos] & 0xf);
            if (next_nclen == 0xf) {
                do {
                    next_nclen += input[++ipos];
                } while (input[ipos] == 0xff);
            }
            
            memcpy(output + opos, input + (++ipos), next_nclen);

            ipos += next_nclen;
            opos += next_nclen;
            if (ipos >= input_size || opos >= output_size) {
                break;
            }
            back_offset = input[ipos] + (input[ipos + 1] << 8);       
            ipos += 2;
            if (next_clen == 0xf+4) {
                do {
                    next_clen += input[ipos++];
                } while (input[ipos - 1] == 0xff);
            }

            if (next_clen <= back_offset) {
                memcpy(output + opos, output + opos - back_offset, next_clen);
            } else {
                memcpy(output + opos, output + opos - back_offset, back_offset);
                for (size_t i = back_offset; i < next_clen; i++) {
                    output[opos + i] = output[opos + i - back_offset];
                }
            }

            opos += next_clen;
        }

        return opos;
    }

    /*!
     *  \brief  extract kernel infos from a given fatbin text section
     *  \param  memory          pointer to the target fatbin text section
     *  \param  memsize         size of the given fatbin text section
     *  \param  desps           vector to store the extracted function metadata
     *  \param  cached_desp_map map of cached function metadata
     */
    static pos_retval_t __extract_kernel_infos(
        void* memory, size_t memsize, std::vector<POSCudaFunctionDesp*>* desps,
        std::map<std::string, POSCudaFunctionDesp*>& cached_desp_map
    ){
        /* =================== ELF utility functions =================== */

        /*!
         *  \brief  identify whether the elf is valid
         *  \param  elf the ELF descriptor
         *  \return POS_SUCCESS for valid;
         *          POS_FAILED for invalid
         */
        auto check_elf = [](Elf *elf) -> pos_retval_t {
            Elf_Kind ek;
            GElf_Ehdr ehdr;
            int elfclass;
            char *id;
            size_t program_header_num;
            size_t sections_num;
            size_t section_str_num;
            
            // verify ELF version
            if((ek=elf_kind(elf)) != ELF_K_ELF){ return POS_FAILED; }

            // obtain ELF header
            if(gelf_getehdr(elf, &ehdr) == NULL){ return POS_FAILED; }

            // verify ELF class
            if ((elfclass = gelf_getclass(elf)) == ELFCLASSNONE){ return POS_FAILED; }

            // verify ELF identification data
            if((id = elf_getident(elf, NULL)) == NULL){ return POS_FAILED; }

            // get the number of section within the ELF
            if (elf_getshdrnum(elf, &sections_num) != 0){ return POS_FAILED; }

            // get the number of program header within the ELF
            // if (elf_getphdrnum(elf, &program_header_num) != 0){ return POS_FAILED; }

            // get the section index of the section header table within the ELF
            if(elf_getshdrstrndx(elf, &section_str_num) != 0){ return POS_FAILED; }

            return POS_SUCCESS;
        };

        /*!
         *  \brief  find the section with the specified name within the ELF
         *  \param  elf     the target ELF
         *  \param  name    name of the section to be found
         *  \param  section pointer to the founded section
         *  \return POS_SUCCESS for successfully founded;
         *          POS_FAILED for internal errors;
         *          POS_FAILED_NOT_EXIST for no section founded
         */
        auto get_section_by_name = [](Elf *elf, const char *name, Elf_Scn **section) -> pos_retval_t {
            Elf_Scn *scn = NULL;
            GElf_Shdr shdr;
            char *section_name = NULL;
            size_t str_section_index;

            POS_CHECK_POINTER(elf); POS_CHECK_POINTER(name); POS_CHECK_POINTER(section);

            if (elf_getshdrstrndx(elf, &str_section_index) != 0){ return POS_FAILED; }

             while ((scn = elf_nextscn(elf, scn)) != NULL) {
                if (gelf_getshdr(scn, &shdr) != &shdr) { return POS_FAILED; }
                if ((section_name = elf_strptr(elf, str_section_index, shdr.sh_name)) == NULL) {
                    return POS_FAILED;
                }
                if (strcmp(section_name, name) == 0) {
                    *section = scn;
                    return POS_SUCCESS;
                }
            }

            return POS_FAILED_NOT_EXIST;
        };
        
        /*!
         *  \brief  extract the symbol table section within the ELF
         *  \param  elf                 the target ELF
         *  \param  symbol_table_data   pointer to the memory to store the extracted section data
         *  \param  symbol_table_size   pointer to the size_t variable to store the size of the symbol table section
         *  \param  symbol_table_shdr   pointer to the memory to store the symbol table section header
         *  \return POS_SUCCESS for successfully extraction;
         *          POS_FAILED for internal errors;
         *          POS_FAILED_NOT_EXIST for no section founded
         */
        auto get_symtab = [&](
            Elf *elf, Elf_Data **symbol_table_data, size_t *symbol_table_size, GElf_Shdr *symbol_table_shdr
        ) -> pos_retval_t
        {
            pos_retval_t retval = POS_SUCCESS;
            GElf_Shdr shdr;
            Elf_Scn *section = NULL;

            POS_CHECK_POINTER(elf); POS_CHECK_POINTER(symbol_table_data); POS_CHECK_POINTER(symbol_table_size); 

            retval = get_section_by_name(elf, ".symtab", &section);
            if(unlikely(retval != POS_SUCCESS)) { return retval; }

            if(gelf_getshdr(section, &shdr) == NULL){ return POS_FAILED; }
            if (symbol_table_shdr != NULL) {
                *symbol_table_shdr = shdr;
            }

            if(shdr.sh_type != SHT_SYMTAB){ return POS_FAILED; }

            if ((*symbol_table_data = elf_getdata(section, NULL)) == NULL) { return POS_FAILED; }
            *symbol_table_size = shdr.sh_size / shdr.sh_entsize;

            return retval;
        };

        /*!
         *  \brief  obtain the ELF section name which contains specified kernel, based on the kernel name (from symbol table)
         *  \param  kernel_name kernel name
         *  \return name of the section to be founded
         */
        auto get_kernel_section_name_from_kernel_name = [](const char *kernel_name) -> char* {
            char *section_name = NULL;

            POS_CHECK_POINTER(kernel_name);

            if (kernel_name[0] == '$') {
                const char *p;
                if ((p = strchr(kernel_name+1, '$')) == NULL) { return NULL; }
                int len = (p - kernel_name) - 1;
                if (asprintf(&section_name, ".nv.info.%.*s", len, kernel_name+1) == -1) { return NULL; }
            } else {
                if (asprintf(&section_name, ".nv.info.%s", kernel_name) == -1) { return NULL; }
            }
            return section_name;
        };

        /*!
         *  \brief  extract parameter info of the kernel within the ELF
         *  \param  elf             descriptor of target ELF file
         *  \param  function_desp   pointer to the pointer of function descriptor
         *  \param  memory          data area of target ELF file
         *  \param  memsize         size of the data area of target ELF file
         *  \return POS_SUCCESS for successfully extraction
         *          POS_FAILED for failed extraction
         *          POS_FAILED_NOT_EXIST for no section was founded
         */
        auto get_params_for_kernel = [&](
            Elf *elf, POSCudaFunctionDesp** function_desp, void* memory, size_t memsize
        ) -> pos_retval_t {
            char *section_name = NULL;
            Elf_Scn *section = NULL;
            Elf_Data *data = NULL;
            size_t secpos=0;

            POS_CHECK_POINTER(elf); POS_CHECK_POINTER(function_desp); POS_CHECK_POINTER(memory);

            // obtain the section that contains the kernel
            if ((section_name = get_kernel_section_name_from_kernel_name((*function_desp)->name.c_str())) == NULL) { 
                POS_WARN("failed to form section name based on kernel's name: kernel_name(%s)", (*function_desp)->name.c_str());
                return POS_FAILED_NOT_EXIST; 
            }
            if (get_section_by_name(elf, section_name, &section) != 0) {
                POS_WARN(
                    "failed to get section based on kernel's name: kernel_name(%s), section_name(%s)",
                    (*function_desp)->name.c_str(), section_name
                );
                return POS_FAILED;
            }
            if ((data = elf_getdata(section, NULL)) == NULL) {
                POS_WARN(
                    "failed to get kernel section data: kernel_name(%s), section_name(%s)",
                    (*function_desp)->name.c_str(), section_name
                );
                return POS_FAILED;
            }

            while (secpos < data->d_size) {
                struct nv_info_kernel_entry *entry = (struct nv_info_kernel_entry*)(data->d_buf+secpos);
                if (entry->format == EIFMT_SVAL && entry->attribute == EIATTR_KPARAM_INFO){
                    if (entry->values_size != 0xc) {
                        POS_WARN("EIATTR_KPARAM_INFO values size has not the expected value of 0xc");
                        return POS_FAILED;
                    }
                    struct nv_info_kparam_info *kparam = (struct nv_info_kparam_info*)&entry->values;
                    (*function_desp)->nb_params += 1;
                    (*function_desp)->param_offsets.push_back(kparam->offset);
                    (*function_desp)->param_sizes.push_back(kparam->size);

                    /*!
                     *  \note   for those parameters that larger that 64, we suspect it might contains device pointer,
                     *          and we will conduct checking when user first launch this kernel
                     */
                    if(unlikely(kparam->size >= 8)){
                        (*function_desp)->suspicious_params.push_back(kparam->ordinal);
                    }

                    secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
                } else if (entry->format == EIFMT_HVAL && entry->attribute == EIATTR_CBANK_PARAM_SIZE) {
                    (*function_desp)->cbank_param_size = entry->values_size;
                    secpos += sizeof(struct nv_info_kernel_entry)-4;
                } else if (entry->format == EIFMT_HVAL) {
                    secpos += sizeof(struct nv_info_kernel_entry)-4;
                } else if (entry->format == EIFMT_SVAL) {
                    secpos += sizeof(struct nv_info_kernel_entry) + entry->values_size-4;
                } else if (entry->format == EIFMT_NVAL) {
                    secpos += sizeof(struct nv_info_kernel_entry)-4;
                } else {
                    secpos += sizeof(struct nv_info_kernel_entry)-4;
                }
            }

            std::reverse((*function_desp)->param_offsets.begin(), (*function_desp)->param_offsets.end());
            std::reverse((*function_desp)->param_sizes.begin(), (*function_desp)->param_sizes.end());

            return POS_SUCCESS;
        };

        /* =============== end of ELF utility functions ================ */

        pos_retval_t retval = POS_SUCCESS, tmp_retval;
        Elf *elf = NULL;
        Elf_Scn *section = NULL;
        Elf_Data *data = NULL, *symbol_table_data = NULL;
        GElf_Shdr symtab_shdr;
        size_t symnum;
        int i = 0, j;
        GElf_Sym sym;
        const char *kernel_str;
        POSCudaFunctionDesp *function_desp;
        bool is_duplicated;

        POS_CHECK_POINTER(memory); POS_CHECK_POINTER(desps);
        POS_ASSERT(memsize > 0);

        // create elf descriptor for the memory region
        POS_CHECK_POINTER(elf = elf_memory((char*)memory, memsize));

        retval = check_elf(elf);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN_DETAIL("invalid ELF format detected");
            goto exit;
        }

        retval = get_symtab(elf, &symbol_table_data, &symnum, &symtab_shdr);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN_DETAIL("failed to extract symbol table section from the ELF");
            goto exit;
        }

        tmp_retval = get_section_by_name(elf, ".nv.info", &section);
        if(unlikely(tmp_retval != POS_SUCCESS)){
            // POS_WARN_DETAIL("failed to obtain section \".nv.info\" in the ELF");
            goto exit;
        }

        if((data = elf_getdata(section, NULL)) == NULL) {
            POS_WARN_DETAIL("failed to obtain data from \".nv.info\" in the ELF");
            goto exit;
        }

        // analyse all kernels within this section
        for (size_t secpos=0; secpos < data->d_size; secpos += sizeof(struct nv_info_entry)){
            struct nv_info_entry *entry = (struct nv_info_entry *)(data->d_buf+secpos);

            if (entry->values_size != 8) {
                POS_WARN(
                    "unexpected values_size of entry within \".nv.info\" section: given(%#x), expected(0)",
                    entry->values_size
                );
                continue;
            }

            // don't add this!
            // if (entry->attribute != EIATTR_FRAME_SIZE) {
            //     POS_WARN(
            //         "unexpected attribute of entry within \".nv.info\" section: given(%#x), expected(%#x)",
            //         entry->attribute, EIATTR_FRAME_SIZE
            //     );
            //     continue;
            // }

            if (entry->kernel_id >= symnum) {
                POS_WARN(
                    "kernel_id out of bounds within \".nv.info\" section: given(%#x), max(%#x)",
                    entry->kernel_id, symnum
                );
                continue;
            }

            // obtain the kernel name from symbol table by given symbol index (i.e., kernel_id)
            if (gelf_getsym(symbol_table_data, entry->kernel_id, &sym) == NULL) {
                POS_WARN(
                    "failed to gelf_getsym the kernel name: kernel_id(%d)",
                    entry->kernel_id
                );
                continue;
            }
            if((kernel_str = elf_strptr(elf, symtab_shdr.sh_link, sym.st_name) ) == NULL){
                POS_WARN(
                    "strptr failed for entry %d", entry->kernel_id
                );
                continue;
            }
            
            /*!
             *  \note   make sure no kernels with the same name are recorded at the same time, those
             *          kernels with same name are the same definition under different PTX/SASS version,
             *          we don't care about the architecture thing under POS, so we just ignore duplication
             */
            is_duplicated = false;
            for(j=0; j<desps->size(); j++){
                if(unlikely(!strcmp(kernel_str, (*desps)[j]->name.c_str()))){
                    is_duplicated = true;
                    break;
                }
            }
            if(unlikely(is_duplicated)){ continue; }

            // check whether this function is cached
            if(likely(cached_desp_map.count(std::string(kernel_str)) > 0)){
                function_desp = cached_desp_map[std::string(kernel_str)];
                desps->push_back(function_desp);
            } else {
                /*!
                 *  \note   we can skip those kernels that won't be called
                 */
                if(cached_desp_map.size() > 0){
                    continue;
                    POS_WARN("found uncached kernels while given cached kernel meta, this might cause nsys to crash: device_name(%s)", kernel_str);
                }

                function_desp = new POSCudaFunctionDesp_t();
                POS_CHECK_POINTER(function_desp);

                function_desp->name = std::string(kernel_str);

                // analyse the parameters of the kernel
                retval = get_params_for_kernel(elf, &function_desp, memory, memsize);
                if(unlikely(retval != POS_SUCCESS)){
                    POS_WARN_DETAIL("failed to extract parameter out of the kernel in the ELF: kernel_name(%s)", kernel_str);
                    goto exit;
                }

                // parsing the parameters hints (e.g., whether it's a pointer, direction of the pointer)
                retval = POSUtil_CUDA_Kernel_Parser::parse_by_prototype(kernel_str, function_desp);
                if(unlikely(retval != POS_SUCCESS)){
                    POS_WARN_DETAIL("failed to extract parameter hints (pointer, direction): kernel_name(%s), won't be recorded!", kernel_str);
                    continue;
                }

                desps->push_back(function_desp);
            }
        }

    exit:
        return retval;
    }
};
