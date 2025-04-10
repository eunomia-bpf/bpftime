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

#include <iostream>


#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/utils/command_caller.h"
#include "pos/include/utils/string.h"
#include "pos/cuda_impl/utils/fatbin.h"


#include <clang-c/Index.h>


/*!
 *  \brief  preprocess a raw demangles name
 *  \param  kernel_str              the raw demangles name
 *  \param  kernel_demangles_name   the processed demangles name
 *  \return POS_SUCCESS for successfully processed
 *          POS_INVALID_INPUT for invalid raw demangles name
 */
pos_retval_t POSUtil_CUDA_Kernel_Parser::__preprocess_prototype(const char *kernel_str, std::string& kernel_demangles_name){
    pos_retval_t retval = POS_SUCCESS;
    size_t dollar_1_pos, dollar_2_pos;

    kernel_demangles_name.clear();

    if(unlikely(strlen(kernel_str) == 0)){
        POS_WARN("kernel demangles name with 0 size provided");
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    /*!
     *  \note   we need to deal with the situation like:
     *          $_ZN2at6native19triu_indices_kernelIlEEvPT_lllll$__cuda_sm20_dsqrt_rn_f64_mediumpath_v1,
     *          we will extract it as _ZN2at6native19triu_indices_kernelIlEEvPT_lllll        
     */
    dollar_1_pos = std::string(kernel_str).find('$');
    dollar_2_pos = std::string(kernel_str).find('$', dollar_1_pos+1);
    if (dollar_1_pos != std::string::npos && dollar_2_pos != std::string::npos){
        kernel_demangles_name = std::string(kernel_str).substr(dollar_1_pos+1, dollar_2_pos-dollar_1_pos-1);
    } else {
        kernel_demangles_name = std::string(kernel_str);
    }

exit:
    return retval;
}


/*!
 *  \brief  generate the origin kernel prototype based on processed demangles name
 *  \param  kernel_demangles_name   the processed demangles name
 *  \param  kernel_prototype        the generated kernel prototype
 *  \return POS_SUCCESS for successfully generation
 *          POS_FAILED for failed generation
 */
pos_retval_t POSUtil_CUDA_Kernel_Parser::__generate_prototype(const std::string& kernel_demangles_name, std::string& kernel_prototype){
    pos_retval_t retval = POS_SUCCESS;
    std::string cmd;

    cmd = std::string("cu++filt ") + kernel_demangles_name;
    kernel_prototype.clear();
    retval = POSUtil_Command_Caller::exec_sync(
        cmd,
        kernel_prototype,
        /* ignore_error */ false,
        /* print_stdout */ false,
        /* print_stderr */ false
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN(
            "failed to analyse paramter details of kernel %s due to failed cu++filt execution: cmd(%s)",
            kernel_demangles_name.c_str(), cmd.c_str()
        );
        goto exit;
    }

exit:
    return retval;
}

/*!
 *  \brief  parsing the kernel prototype
 *  \param  kernel_prototype        the generated kernel prototype
 *  \param  function_desp           pointer to the function descriptor
 *  \return POS_SUCCESS for successfully processed
 *          POS_FAILED for failed processed
 */
pos_retval_t POSUtil_CUDA_Kernel_Parser::__parse_prototype(const std::string& kernel_prototype, POSCudaFunctionDesp *function_desp){
    pos_retval_t retval = POS_SUCCESS;
    CXIndex index;
    CXErrorCode cx_retval;
    CXUnsavedFile unsaved_file;
    CXTranslationUnit translation_unit;
    CXCursor root_cursor;
    uint64_t i;
    std::string cast_kernel_prototype;

    typedef struct __visit_meta {
        POSCudaFunctionDesp *function_desp;
        uint32_t param_index;
    };
    
    __visit_meta vm;
    vm.function_desp = function_desp;
    vm.param_index = 0;

    /*!
     *  \note   we need to cast the demangle name from 'cu++filt' to valid c++ function signature, so that we can use clang to parse its
     *          parameter list
     */
    auto form_valid_prototype = [](const std::string &prototype) -> std::string {
        std::string mock_prototype, param_list;
        pos_retval_t pos_tmp_retval;

        // extract the parameter list
        pos_tmp_retval = POSUtil_String::extract_substring_from_field</* reverse */true>('(', ')', prototype, param_list);
        if(unlikely(pos_tmp_retval != POS_SUCCESS)){
            mock_prototype = std::string("void mocked_func();");
        } else {
            mock_prototype = std::string("void mocked_func") + param_list + std::string(";");
        }

        return mock_prototype;
    };
    cast_kernel_prototype = form_valid_prototype(kernel_prototype);
    
    // create clang index
    index = clang_createIndex(0, 0);

    // create clang translation unit
    unsaved_file.Filename = "temp.cpp";
    unsaved_file.Contents = cast_kernel_prototype.c_str();
    unsaved_file.Length = static_cast<unsigned long>(cast_kernel_prototype.length());
    cx_retval = clang_parseTranslationUnit2(
        /* CIdx */ index,
        /* source_filename */ "temp.cpp",
        /* command_line_args */ nullptr,
        /* nb_command_line_args */ 0,
        /* unsaved_files */ &unsaved_file,
        /* nb_unsaved_file */ 1,
        /* options */ CXTranslationUnit_None,
        /* out_TU */ &translation_unit
    );
    if(unlikely(cx_retval != CXError_Success)){
        POS_WARN_DETAIL("failed to parse the function prototype from the memory buffer");
        retval = POS_FAILED;
        goto exit;
    }
    if(unlikely(translation_unit == nullptr)){
        POS_ERROR_DETAIL("failed to create clang translation unit");
    }

    // traverse parameter list
    root_cursor = clang_getTranslationUnitCursor(translation_unit);
    clang_visitChildren(
        /* parent */ root_cursor,
        /* visitor */ 
        [](CXCursor cursor, CXCursor parent, CXClientData clientData) -> CXChildVisitResult {
            CXType type = clang_getCursorType(cursor);
            CXString typeName = clang_getTypeSpelling(type);
            CXType pointeeType;
            __visit_meta *vm;
            
            POS_CHECK_POINTER(vm = (__visit_meta*)(clientData));

            if(cursor.kind == CXCursor_ParmDecl) { // parameter 
                if (type.kind == CXType_Pointer) { // pointer type
                    pointeeType = clang_getPointeeType(type);

                    if(clang_isConstQualifiedType(pointeeType)){ // constant pointer type
                        vm->function_desp->input_pointer_params.push_back(vm->param_index);
                    } else {
                        /*!
                         *  \note   for non-const pointers, we need to classify them as non-const
                         *          pointers, as they might also be read by the kernel
                         */
                        vm->function_desp->inout_pointer_params.push_back(vm->param_index);
                    }
                }
                vm->param_index += 1;
            }

            return CXChildVisit_Recurse;
        },
        /* client_data */ &vm
    );

    clang_disposeTranslationUnit(translation_unit);
    clang_disposeIndex(index);

    // POS_LOG("cu++filt result: %s", kernel_prototype.c_str());
    // POS_LOG("input pointers:");
    // for(i=0; i<function_desp->input_pointer_params.size(); i++){ POS_LOG("    %lu", function_desp->input_pointer_params[i]); }
    // POS_LOG("output pointers:");
    // for(i=0; i<function_desp->output_pointer_params.size(); i++){ POS_LOG("    %lu", function_desp->output_pointer_params[i]); }

exit:
    return retval;
}
