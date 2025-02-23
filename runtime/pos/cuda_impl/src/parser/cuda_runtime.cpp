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
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/api_context.h"
#include "pos/cuda_impl/utils/fatbin.h"

namespace ps_functions {


/*!
 *  \related    cudaMalloc
 *  \brief      allocate a memory area
 */
namespace cuda_malloc {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandleManager_CUDA_Context *hm_context;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

    #if POS_CONF_RUNTIME_EnableDebugCheck
        // check whether given parameter is valid
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_malloc): failed to parse cuda_malloc, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Context, POSHandleManager_CUDA_Context
        );
        POS_CHECK_POINTER(hm_context);
        POS_CHECK_POINTER(hm_context->latest_used_handle);

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ hm_context->latest_used_handle
        });

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // operate on handler manager
        retval = hm_memory->allocate_mocked_resource(
            /* handle */ &memory_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Context, 
                /* handles */ std::vector<POSHandle*>({hm_context->latest_used_handle}) 
            }}),
            /* size */ pos_api_param_value(wqe, 0, size_t),
            /* use_expected_addr */ false,
            /* expected_addr */ 0,
            /* state_size */ (uint64_t)pos_api_param_value(wqe, 0, size_t)
        );

        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cuda_malloc): failed to allocate mocked resource within the CUDA memory handler manager");
            memset(wqe->api_cxt->ret_data, 0, sizeof(uint64_t));
            goto exit;
        } else {
            memcpy(wqe->api_cxt->ret_data, &(memory_handle->client_addr), sizeof(uint64_t));
        }
        
        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ memory_handle
        });

    exit:
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;
        return retval;
    }

} // namespace cuda_malloc


/*!
 *  \related    cudaFree
 *  \brief      release a CUDA memory area
 */
namespace cuda_free {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_free): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // operate on handler manager
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, void*),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            // POS_WARN(
            //     "parse(cuda_free): no CUDA memory was founded: client_addr(%p)",
            //     (void*)pos_api_param_value(wqe, 0, void*)
            // );
            goto exit;
        }

        memory_handle->mark_status(kPOS_HandleStatus_Delete_Pending);

        wqe->record_handle<kPOS_Edge_Direction_Delete>({
            /* handle */ memory_handle
        });
        
    exit:
        return retval;
    }
} // namespace cuda_free


/*!
 *  \related    cudaLaunchKernel
 *  \brief      launch a user-define computation kernel
 */
namespace cuda_launch_kernel {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;
        POSClient_CUDA *client;
        POSHandle_CUDA_Function *function_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandle_CUDA_Memory *memory_handle;

        uint64_t i, j, param_index;
        void *args, *arg_addr, *arg_value;

        uint8_t *struct_base_ptr;
        uint64_t arg_size, struct_offset;

        POSHandleManager_CUDA_Function *hm_function;
        POSHandleManager_CUDA_Stream *hm_stream;
        POSHandleManager_CUDA_Memory *hm_memory;

        /*!
         *  \brief  obtain a potential pointer from a struct by given offset within the struct
         *  \param  base    base address of the struct
         *  \param  offset  offset within the struct
         *  \return potential pointer
         */
        auto __try_get_potential_addr_from_struct_with_offset = [](uint8_t* base, uint64_t offset) -> void* {
            uint8_t *bias_base = base + offset;
            POS_CHECK_POINTER(bias_base);

        #define __ADDR_UNIT(index)   ((uint64_t)(*(bias_base+index) & 0xff) << (index*8))
            return (void*)(
                __ADDR_UNIT(0) | __ADDR_UNIT(1) | __ADDR_UNIT(2) | __ADDR_UNIT(3) | __ADDR_UNIT(4) | __ADDR_UNIT(5)
            );
        #undef __ADDR_UNIT
        };

        /*!
         *  \brief  printing the kernels direction after first parsing
         *  \param  function_handle handler of the function to be printed
         */
        auto __print_kernel_directions = [](POSHandle_CUDA_Function *function_handle){
            POS_CHECK_POINTER(function_handle);
            POS_LOG("obtained direction of kernel %s:", function_handle->signature.c_str());

            // for printing input / output
            auto __unit_print_input_output = [](std::vector<uint32_t>& vec, const char* dir_string){
                uint64_t i, param_index;
                static char param_idx[2048] = {0};
                memset(param_idx, 0, sizeof(param_idx));
                for(i=0; i<vec.size(); i++){
                    param_index = vec[i];
                    if(likely(i!=0)){
                        sprintf(param_idx, "%s, %lu", param_idx, param_index); 
                    } else {
                        sprintf(param_idx, "%lu", param_index);
                    }
                }
                POS_LOG("    %s params: %s", dir_string, param_idx);
            };

            // for printing inout
            auto __unit_print_inout = [](std::vector<std::pair<uint32_t, uint64_t>>& vec, const char* dir_string){
                uint64_t i, struct_offset, param_index;
                static char param_idx[2048] = {0};
                memset(param_idx, 0, sizeof(param_idx));
                for(i=0; i<vec.size(); i++){
                    param_index = vec[i].first;
                    struct_offset = vec[i].second;
                    if(likely(i != 0)){
                        sprintf(param_idx, "%s, %lu(ofs: %lu)", param_idx, param_index, struct_offset); 
                    } else {
                        sprintf(param_idx, "%lu(ofs: %lu)", param_index, struct_offset);
                    }
                };
                POS_LOG("    %s params: %s", dir_string, param_idx);
            };

            __unit_print_input_output(function_handle->input_pointer_params, "input");
            __unit_print_input_output(function_handle->output_pointer_params, "output");
            __unit_print_inout(function_handle->confirmed_suspicious_params, "inout");
        };

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);
        
        // check whether given parameter is valid
        #if POS_CONF_RUNTIME_EnableDebugCheck
            if(unlikely(wqe->api_cxt->params.size() != 6)){
                POS_WARN(
                    "parse(cuda_launch_kernel): failed to parse, given %lu params, %lu expected",
                    wqe->api_cxt->params.size(), 6
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
        #endif

        // obtain handle managers of function, stream and memory
        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // find out the involved function
        retval = hm_function->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &function_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_launch_kernel): no function was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ function_handle
        });

        // find out the involved stream
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 5, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_launch_kernel): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 5, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ stream_handle
        });

        // the 3rd parameter of the API call contains parameter to launch the kernel
        args = pos_api_param_addr(wqe, 3);
        POS_CHECK_POINTER(args);

        // [Cricket Adapt] skip the metadata used by cricket
        args += (sizeof(size_t) + sizeof(uint16_t) * function_handle->nb_params);
        
        /*!
         *  \note   record all input memory areas
         */
        for(i=0; i<function_handle->input_pointer_params.size(); i++){
            param_index = function_handle->input_pointer_params[i];
 
            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as input during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });
        }
        
        /*!
         *  \note   record all inout memory areas
         */
        for(i=0; i<function_handle->inout_pointer_params.size(); i++){
            param_index = function_handle->inout_pointer_params[i];

            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as inout during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_InOut>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });

            hm_memory->record_modified_handle(memory_handle);
        }

        /*!
         *  \note   record all output memory areas
         */
        for(i=0; i<function_handle->output_pointer_params.size(); i++){
            param_index = function_handle->output_pointer_params[i];

            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as output during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_Out>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });

            hm_memory->record_modified_handle(memory_handle);
        }

        /*!
         *  \note   check suspicious parameters that might contains pointer
         *  \warn   only check once?
         */
        if(unlikely(function_handle->has_verified_params == false)){          
            for(i=0; i<function_handle->suspicious_params.size(); i++){
                param_index = function_handle->suspicious_params[i];

                // we can skip those already be identified as input / output
                if(std::find(
                    function_handle->input_pointer_params.begin(),
                    function_handle->input_pointer_params.end(),
                    param_index
                ) != function_handle->input_pointer_params.end()){
                    continue;
                }
                if(std::find(
                    function_handle->output_pointer_params.begin(),
                    function_handle->output_pointer_params.end(),
                    param_index
                ) != function_handle->output_pointer_params.end()){
                    continue;
                }

                arg_addr = args + function_handle->param_offsets[param_index];
                POS_CHECK_POINTER(arg_addr);

                struct_base_ptr = (uint8_t*)arg_addr;

                arg_size = function_handle->param_sizes[param_index];
                POS_ASSERT(arg_size >= 6);

                // iterate across the struct using a 8-bytes window
                for(j=0; j<arg_size-6; j++){
                    arg_value = __try_get_potential_addr_from_struct_with_offset(struct_base_ptr, j);

                    tmp_retval = hm_memory->get_handle_by_client_addr(
                        /* client_addr */ arg_value,
                        /* handle */ &memory_handle
                    );
                    if(unlikely(tmp_retval == POS_SUCCESS)){
                        // we treat such memory areas as inout memory
                        function_handle->confirmed_suspicious_params.push_back({
                            /* parameter index */ param_index,
                            /* offset */ j  
                        });

                        wqe->record_handle<kPOS_Edge_Direction_InOut>({
                            /* handle */ memory_handle,
                            /* param_index */ param_index,
                            /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
                        });

                        hm_memory->record_modified_handle(memory_handle);
                    }
                } // foreach arg_size
            } // foreach suspicious_params

            function_handle->has_verified_params = true;
            
            // __print_kernel_directions(function_handle);
        } else {
            for(i=0; i<function_handle->confirmed_suspicious_params.size(); i++){
                param_index = function_handle->confirmed_suspicious_params[i].first;
                struct_offset = function_handle->confirmed_suspicious_params[i].second;

                arg_addr = args + function_handle->param_offsets[param_index];
                POS_CHECK_POINTER(arg_addr);
                arg_value = *((void**)(arg_addr+struct_offset));

                /*!
                 *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
                 *          this is probably normal, so we just ignore this situation
                 */
                if(unlikely(arg_value == nullptr)){
                    continue;
                }

                tmp_retval = hm_memory->get_handle_by_client_addr(
                    /* client_addr */ arg_value,
                    /* handle */ &memory_handle
                );

                if(unlikely(tmp_retval != POS_SUCCESS)){
                    // POS_WARN(
                    //     "%lu(th) parameter of kernel %s is marked as suspicious output during kernel parsing phrase, "
                    //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                    //     param_index, function_handle->signature.c_str(), arg_value
                    // );
                    continue;
                }

                wqe->record_handle<kPOS_Edge_Direction_InOut>({
                    /* handle */ memory_handle,
                    /* param_index */ param_index,
                    /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
                });

                hm_memory->record_modified_handle(memory_handle);
            }
        }

    #if POS_CONF_RUNTIME_EnableTrace
        parser->metric_reducers.reduce(
            /* index */ POSParser::KERNEL_in_memories,
            /* value */ function_handle->input_pointer_params.size()
                        + function_handle->inout_pointer_params.size()
        );
        parser->metric_reducers.reduce(
            /* index */ POSParser::KERNEL_out_memories,
            /* value */ function_handle->output_pointer_params.size()
                        + function_handle->inout_pointer_params.size()
        );
        parser->metric_counters.add_counter(
            /* index */ POSParser::KERNEL_number_of_user_kernels
        );
    #endif

    #if POS_PRINT_DEBUG
        typedef struct __dim3 { uint32_t x; uint32_t y; uint32_t z; } __dim3_t;
        POS_DEBUG(
            "parse(cuda_launch_kernel): function(%s), stream(%p), grid_dim(%u,%u,%u), block_dim(%u,%u,%u), SM_size(%lu)",
            function_handle->name.c_str(), stream_handle->server_addr,
            ((__dim3_t*)pos_api_param_addr(wqe, 1))->x,
            ((__dim3_t*)pos_api_param_addr(wqe, 1))->y,
            ((__dim3_t*)pos_api_param_addr(wqe, 1))->z,
            ((__dim3_t*)pos_api_param_addr(wqe, 2))->x,
            ((__dim3_t*)pos_api_param_addr(wqe, 2))->y,
            ((__dim3_t*)pos_api_param_addr(wqe, 2))->z,
            pos_api_param_value(wqe, 4, size_t)
        );
    #endif

    exit:
        return retval;
    }

} // namespace cuda_launch_kernel




/*!
 *  \related    cudaMemcpy (Host to Device)
 *  \brief      copy memory buffer from host to device
 */
namespace cuda_memcpy_h2d {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 2)){
            POS_WARN(
                "parse(cuda_memcpy_h2d): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 2
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // try obtain the destination memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_h2d): no memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_InOut>({
                /* handle */ memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle->client_addr)
            });
            hm_memory->record_modified_handle(memory_handle);
        }


    exit:
        return retval;
    }

} // namespace cuda_memcpy_h2d



/*!
 *  \related    cudaMemcpy (Device to Host)
 *  \brief      copy memory buffer from device to host
 */
namespace cuda_memcpy_d2h {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 2)){
            POS_WARN(
                "parse(cuda_memcpy_d2h): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 2
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // try obtain the source memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2h): no memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle->client_addr)
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memcpy_d2h




/*!
 *  \related    cudaMemcpy (Device to Device)
 *  \brief      copy memory buffer from device to device
 */
namespace cuda_memcpy_d2d {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;
        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *dst_memory_handle, *src_memory_handle;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 3)){
            POS_WARN(
                "parse(cuda_memcpy_d2d): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 3
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // try obtain the destination memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &dst_memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d): no destination memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_Out>({
                /* handle */ dst_memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(dst_memory_handle->client_addr)
            });
            hm_memory->record_modified_handle(dst_memory_handle);
        }

        // try obtain the source memory handles
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 1, uint64_t),
            /* handle */ &src_memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d): no source memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 1, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ src_memory_handle,
                /* param_index */ 1,
                /* offset */ pos_api_param_value(wqe, 1, uint64_t) - (uint64_t)(src_memory_handle->client_addr)
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memcpy_d2d




/*!
 *  \related    cudaMemcpyAsync (Host to Device)
 *  \brief      async copy memory buffer from host to device
 */
namespace cuda_memcpy_h2d_async {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 3)){
            POS_WARN(
                "parse(cuda_memcpy_h2d_async): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 3
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the destination memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_h2d_async): no memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_InOut>({
                /* handle */ memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle->client_addr)
            });
            hm_memory->record_modified_handle(memory_handle);
        }

        // try obtain the stream handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 2, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_h2d_async): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 2, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ stream_handle
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memcpy_h2d_async




/*!
 *  \related    cudaMemcpyAsync (Device to Host)
 *  \brief      async copy memory buffer from device to host
 */
namespace cuda_memcpy_d2h_async {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 3)){
            POS_WARN(
                "parse(cuda_memcpy_d2h_async): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 3
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the source memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2h_async): no memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle->client_addr)
            });
        }

        // try obtain the stream handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 2, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2h_async): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 2, uint64_t)
            );
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ stream_handle
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memcpy_d2h_async




/*!
 *  \related    cudaMemcpyAsync (Device to Device)
 *  \brief      async copy memory buffer from device to device
 */
namespace cuda_memcpy_d2d_async {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *dst_memory_handle, *src_memory_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 4)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 4
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the destination memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &dst_memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): no destination memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_Out>({
                /* handle */ dst_memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(dst_memory_handle->client_addr)
            });
            hm_memory->record_modified_handle(dst_memory_handle);
        }

        // try obtain the source memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 1, uint64_t),
            /* handle */ &src_memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): no source memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 1, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ src_memory_handle,
                /* param_index */ 1,
                /* offset */ pos_api_param_value(wqe, 1, uint64_t) - (uint64_t)(src_memory_handle->client_addr)
            });
        }

        // try obtain the stream handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 3, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 3, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ stream_handle
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memcpy_d2d_async



/*!
 *  \related    cudaMemsetAsync
 *  \brief      async set memory area to a specific value
 */
namespace cuda_memset_async {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;

        POSClient_CUDA *client;
        POSHandle_CUDA_Memory *memory_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 4)){
            POS_WARN(
                "parse(cuda_memset_async): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 4
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the destination memory handle
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &memory_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): no destination memory was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_Out>({
                /* handle */ memory_handle,
                /* param_index */ 0,
                /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle->client_addr)
            });
            hm_memory->record_modified_handle(memory_handle);
        }

        // try obtain the stream handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 3, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 3, uint64_t)
            );
            goto exit;
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ stream_handle
            });
        }

    exit:
        return retval;
    }

} // namespace cuda_memset_async



/*!
 *  \related    cudaSetDevice
 *  \brief      specify a CUDA device to use
 */
namespace cuda_set_device {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;

        POSHandleManager_CUDA_Device *hm_device;
        POSHandle_CUDA_Device *device_handle;

        POSHandleManager_CUDA_Context *hm_context;
        POSHandle_CUDA_Context *context_handle;
        uint64_t i, nb_context;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_set_device): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        // obtain handle managers of device
        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);

        // find out the involved device
        retval = hm_device->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, int),
            /* handle */ &device_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_set_device): no device was founded: client_addr(%d)",
                (uint64_t)pos_api_param_value(wqe, 0, int)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ device_handle
        });

        hm_device->latest_used_handle = device_handle;


        // update latest-used context
        hm_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Context, POSHandleManager_CUDA_Context
        );
        POS_CHECK_POINTER(hm_context);

        nb_context = hm_context->get_nb_handles();
        for(i=0; i<nb_context; i++){
            POS_CHECK_POINTER(context_handle = hm_context->get_handle_by_id(i));
            POS_ASSERT(context_handle->parent_handles.size() == 1);
            POS_ASSERT(context_handle->parent_handles[0]->resource_type_id == kPOS_ResourceTypeId_CUDA_Device);
            if(context_handle->parent_handles[0] == device_handle){
                hm_context->latest_used_handle = context_handle;
                break;
            }

            if(unlikely(i == nb_context-1)){
                POS_ERROR_DETAIL("failed to update latest context, no context on device, this is a bug");
            }
        }

    exit:
        return retval;
    }
} // namespace cuda_set_device



/*!
 *  \related    cudaGetLastError
 *  \brief      obtain the latest error within the CUDA context
 */
namespace cuda_get_last_error {
    // parser function
    POS_RT_FUNC_PARSER(){
        return POS_SUCCESS;
    }
} // namespace cuda_get_last_error



/*!
 *  \related    cudaGetErrorString
 *  \brief      obtain the error string from the CUDA context
 */
namespace cuda_get_error_string {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        
        client = (POSClient_CUDA*)(wqe->client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_get_error_string): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

    exit:   
        return retval;
    }

} // namespace cuda_get_error_string


/*!
 *  \related    cudaPeekAtLastError
 *  \brief      obtain the latest error within the CUDA context
 */
namespace cuda_peek_at_last_error {
    // parser function
    POS_RT_FUNC_PARSER(){
        return POS_SUCCESS;
    }
} // namespace cuda_peek_at_last_error



/*!
 *  \related    cudaGetDeviceCount
 *  \brief      obtain the number of devices
 */
namespace cuda_get_device_count {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        uint64_t nb_handles;
        int nb_handles_int;

        POSHandleManager_CUDA_Device *hm_device;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // obtain handle managers of device
        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);

        nb_handles = hm_device->get_nb_handles();
        nb_handles_int = (int)nb_handles;

        POS_CHECK_POINTER(wqe->api_cxt->ret_data);
        memcpy(wqe->api_cxt->ret_data, &nb_handles_int, sizeof(int));

        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

        return retval;
    }
} // namespace cuda_get_device_count




/*!
 *  \related    cudaGetDeviceProperties
 *  \brief      obtain the properties of specified device
 */
namespace cuda_get_device_properties {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;

        POSClient_CUDA *client;
        POSHandle_CUDA_Device *device_handle;
        POSHandleManager_CUDA_Device *hm_device;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_memcpy_d2d_async): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);

        // find out the involved device
        retval = hm_device->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, int),
            /* handle */ &device_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_set_device): no device was founded: client_addr(%d)",
                (uint64_t)pos_api_param_value(wqe, 0, int)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ device_handle
        });

    exit:
        return retval;
    }

} // namespace cuda_get_device_properties



/*!
 *  \related    cudaDeviceGetAttribute
 *  \brief      obtain the properties of specified device
 */
namespace cuda_device_get_attribute {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;

        POSClient_CUDA *client;
        POSHandle_CUDA_Device *device_handle;
        POSHandleManager_CUDA_Device *hm_device;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 2)){
            POS_WARN(
                "parse(cuda_device_get_attribute): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 2
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);

        // find out the involved device
        retval = hm_device->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 1, int),
            /* handle */ &device_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_device_get_attribute): no device was founded: client_addr(%d)",
                (uint64_t)pos_api_param_value(wqe, 1, int)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ device_handle
        });

    exit:
        return retval;
    }

} // namespace cuda_device_get_attribute



/*!
 *  \related    cudaGetDevice
 *  \brief      returns which device is currently being used
 */
namespace cuda_get_device {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandleManager_CUDA_Device *hm_device;
        int latest_device_id;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // obtain handle managers of device
        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);
        POS_CHECK_POINTER(hm_device->latest_used_handle);

        POS_CHECK_POINTER(wqe->api_cxt->ret_data);

        latest_device_id = static_cast<int>((uint64_t)(hm_device->latest_used_handle->client_addr));
        memcpy(wqe->api_cxt->ret_data, &(latest_device_id), sizeof(int));

        // the api is finish, one can directly return
        wqe->status = kPOS_API_Execute_Status_Return_Without_Worker;

    exit:
        return retval;
    }
} // namespace cuda_get_device



/*!
 *  \related    cudaFuncGetAttributes
 *  \brief      find out attributes for a given function
 */
namespace cuda_func_get_attributes {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Function *function_handle;
        POSHandleManager_CUDA_Function *hm_function;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_func_get_attributes): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        // obtain handle managers of device
        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

        // find out the involved function
        retval = hm_function->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &function_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_func_get_attributes): no function was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ function_handle
        });

    exit:
        return retval;
    }
} // namespace cuda_func_get_attributes



/*!
 *  \related    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 *  \brief      returns occupancy for a device function with the specified flags
 */
namespace cuda_occupancy_max_active_bpm_with_flags {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Function *function_handle;
        POSHandleManager_CUDA_Function *hm_function;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 4)){
            POS_WARN(
                "parse(cuda_occupancy_max_active_bpm_with_flags): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 4
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        // obtain handle managers of device
        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

        // find out the involved function
        retval = hm_function->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &function_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_occupancy_max_active_bpm_with_flags): no function was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ function_handle
        });

    exit:
        return retval;
    }
} // namespace cuda_occupancy_max_active_bpm_with_flags



/*!
 *  \related    cudaStreamSynchronize
 *  \brief      sync a specified stream
 */
namespace cuda_stream_synchronize {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_stream_synchronize): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the source memory handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_stream_synchronize): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
        } else {
            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ stream_handle
            });
        }
 
    exit:
        return retval;
    }

} // namespace cuda_stream_synchronize




/*!
 *  \related    cudaStreamIsCapturing
 *  \brief      obtain the stream's capturing state
 */
namespace cuda_stream_is_capturing {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Stream *hm_stream;

        cudaStreamCaptureStatus capture_status;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_stream_is_capturing): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // try obtain the stream handle
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_stream_is_capturing): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }

        // if(likely(stream_handle->is_capturing == false)){
        //     capture_status = cudaStreamCaptureStatusNone;
        // } else {
        //     capture_status = cudaStreamCaptureStatusActive;
        // }
        // memcpy(wqe->api_cxt->ret_data, &capture_status, sizeof(cudaStreamCaptureStatus));
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ stream_handle
        });

        // mark this sync call can be returned after parsing
        // wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    exit:
        return retval;
    }

} // namespace cuda_stream_is_capturing




/*!
 *  \related    cuda_event_create_with_flags
 *  \brief      create cudaEvent_t with flags
 */
namespace cuda_event_create_with_flags {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Event *event_handle;
        POSHandleManager_CUDA_Event *hm_event;
        POSHandleManager_CUDA_Context *hm_context;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cuda_event_create_with_flags): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Context, POSHandleManager_CUDA_Context
        );
        POS_CHECK_POINTER(hm_context);
        POS_CHECK_POINTER(hm_context->latest_used_handle);

        hm_event = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Event, POSHandleManager_CUDA_Event
        );
        POS_CHECK_POINTER(hm_event);

        // operate on handler manager
        retval = hm_event->allocate_mocked_resource(
            /* handle */ &event_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Context, 
                /* handles */ std::vector<POSHandle*>({hm_context->latest_used_handle}) 
            }})
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cuda_event_create_with_flags): failed to allocate mocked resource within the CUDA event handler manager");
            memset(wqe->api_cxt->ret_data, 0, sizeof(cudaEvent_t));
            goto exit;
        }
        
        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ event_handle
        });

        event_handle->flags = pos_api_param_value(wqe, 0, int);

        // mark this sync call can be returned after parsing
        memcpy(wqe->api_cxt->ret_data, &(event_handle->client_addr), sizeof(cudaEvent_t));
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    exit:
        return retval;
    }

} // namespace cuda_event_create_with_flags




/*!
 *  \related    cuda_event_destory
 *  \brief      destory a CUDA event
 */
namespace cuda_event_destory {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Event *event_handle;
        POSHandleManager_CUDA_Event *hm_event;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cublas_set_math_mode): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_event = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Event, POSHandleManager_CUDA_Event
        );
        POS_CHECK_POINTER(hm_event);

        // operate on handler manager
        retval = hm_event->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0,  cudaEvent_t),
            /* handle */ &event_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_event_destory): no CUDA event was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, cudaEvent_t)
            );
            goto exit;
        }
        
        event_handle->mark_status(kPOS_HandleStatus_Delete_Pending);
    
        wqe->record_handle<kPOS_Edge_Direction_Delete>({
            /* handle */ event_handle
        });

    exit:
        return retval;
    }

} // namespace cuda_event_destory




/*!
 *  \related    cuda_event_record
 *  \brief      record a CUDA event
 */
namespace cuda_event_record {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Event *event_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandleManager_CUDA_Event *hm_event;
        POSHandleManager_CUDA_Stream *hm_stream;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 2)){
            POS_WARN(
                "parse(cublas_set_math_mode): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 2
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_event = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Event, POSHandleManager_CUDA_Event
        );
        POS_CHECK_POINTER(hm_event);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // operate on handler manager
        retval = hm_event->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, cudaEvent_t),
            /* handle */ &event_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_event_record): no CUDA event was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, cudaEvent_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_Out>({
            /* handle */ event_handle
        });

        hm_event->record_modified_handle(event_handle);

        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 1, cudaStream_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_event_record): no CUDA stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 1, cudaStream_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ stream_handle
        });

    exit:
        return retval;
    }

} // namespace cuda_event_record



/*!
 *  \related    cudaEventQuery
 *  \brief      query the state of an event
 */
namespace cuda_event_query {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Event *event_handle;
        POSHandleManager_CUDA_Event *hm_event;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cublas_set_math_mode): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_event = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Event, POSHandleManager_CUDA_Event
        );
        POS_CHECK_POINTER(hm_event);

        // operate on handler manager
        retval = hm_event->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, cudaEvent_t),
            /* handle */ &event_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_event_record): no CUDA event was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, cudaEvent_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ event_handle
        });

    exit:
        return retval;
    }

} // namespace cuda_event_query




/*!
 *  \related    template_cuda
 *  \brief      template_cuda
 */
namespace template_cuda {
    // parser function
    POS_RT_FUNC_PARSER(){
        return POS_FAILED_NOT_IMPLEMENTED;
    }

} // namespace template_cuda


} // namespace ps_functions
