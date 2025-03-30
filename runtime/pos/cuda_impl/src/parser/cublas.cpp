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

#include "cublas_v2.h"

#include "pos/include/common.h"

#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/api_context.h"
#include "pos/cuda_impl/utils/fatbin.h"

namespace ps_functions {


/*!
 *  \related    cuBlasCreate
 *  \brief      create a cuBlas context
 */
namespace cublas_create {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_cuBLAS_Context *cublas_context_handle;
        POSHandleManager_CUDA_Stream *hm_stream;
        POSHandleManager_cuBLAS_Context* hm_cublas_context;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        hm_cublas_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_cuBLAS_Context, POSHandleManager_cuBLAS_Context
        );
        POS_CHECK_POINTER(hm_cublas_context);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        // operate on handler manager
        retval = hm_cublas_context->allocate_mocked_resource(
            /* handle */ &cublas_context_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Stream, 
                /* handles */ std::vector<POSHandle*>({hm_stream->default_handle}) 
            }}),
            /* size */ sizeof(cublasHandle_t),
            /* use_expected_addr */ false,
            /* expected_addr */ 0,
            /* state_size */ 0
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cublas_create): failed to allocate mocked cublas context within the handler manager");
            memset(wqe->api_cxt->ret_data, 0, sizeof(cublasHandle_t));
            goto exit;
        }
        POS_DEBUG(
            "parse(cublas_create): allocate mocked cublas context within the handler manager: addr(%p), size(%lu)",
            cublas_context_handle->client_addr, cublas_context_handle->size
        );

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ cublas_context_handle
        });

        // mark this sync call can be returned after parsing
        memcpy(wqe->api_cxt->ret_data, &(cublas_context_handle->client_addr), sizeof(cublasHandle_t));
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    exit:
        return retval;
    }

} // namespace cublas_create


/*!
 *  \related    cuBlasSetStream
 *  \brief      todo
 */
namespace cublas_set_stream {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandle_cuBLAS_Context *cublas_context_handle;
        POSHandleManager_CUDA_Stream *hm_stream;
        POSHandleManager_cuBLAS_Context *hm_cublas_context;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 2)){
            POS_WARN(
                "parse(cuda_malloc): failed to parse cuda_malloc, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 2
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        hm_cublas_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_cuBLAS_Context, POSHandleManager_cuBLAS_Context
        );
        POS_CHECK_POINTER(hm_cublas_context);

        // operate on handler manager
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 1, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_set_stream): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 1, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ stream_handle
        });
        retval = hm_cublas_context->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &cublas_context_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_set_stream): no cuBLAS context was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ cublas_context_handle
        });

    exit:
        return retval;
    }
} // namespace cublas_set_stream




/*!
 *  \related    cuBlasSetMathMode
 *  \brief      todo
 */
namespace cublas_set_math_mode {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_cuBLAS_Context *cublas_context_handle;
        POSHandleManager_cuBLAS_Context *hm_cublas_context;

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

        hm_cublas_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_cuBLAS_Context, POSHandleManager_cuBLAS_Context
        );
        POS_CHECK_POINTER(hm_cublas_context);

        // operate on handler manager
        retval = hm_cublas_context->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &cublas_context_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_set_math_mode): no cuBLAS context was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ cublas_context_handle
        });
        
        
        

    exit:
        return retval;
    }

} // namespace cublas_set_math_mode




/*!
 *  \related    cuBlasSGemm
 *  \brief      todo
 */
namespace cublas_sgemm {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_cuBLAS_Context *cublas_context_handle;
        POSHandle_CUDA_Memory *memory_handle_A, *memory_handle_B, *memory_handle_C;
        POSHandleManager_cuBLAS_Context *hm_cublas_context;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 14)){
            POS_WARN(
                "parse(cublas_sgemm): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 14
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_cublas_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_cuBLAS_Context, POSHandleManager_cuBLAS_Context
        );
        POS_CHECK_POINTER(hm_cublas_context);

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // operate on handler manager
        retval = hm_cublas_context->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &cublas_context_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm): no cuBLAS context was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ cublas_context_handle
        });

        // operate on memory manager
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 7, uint64_t),
            /* handle */ &memory_handle_A
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm): no memory handle A was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 7, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ memory_handle_A,
            /* param_index */ 7,
            /* offset */ pos_api_param_value(wqe, 7, uint64_t) - (uint64_t)(memory_handle_A->client_addr)
        });

        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 9, uint64_t),
            /* handle */ &memory_handle_B
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm): no memory handle B was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 9, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ memory_handle_B,
            /* param_index */ 9,
            /* offset */ pos_api_param_value(wqe, 9, uint64_t) - (uint64_t)(memory_handle_B->client_addr)
        });

        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 12, uint64_t),
            /* handle */ &memory_handle_C
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm): no memory handle C was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 12, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_Out>({
            /* handle */ memory_handle_C,
            /* param_index */ 12,
            /* offset */ pos_api_param_value(wqe, 12, uint64_t) - (uint64_t)(memory_handle_C->client_addr)
        });

        hm_memory->record_modified_handle(memory_handle_C);

        #if POS_CONF_RUNTIME_EnableTrace
            parser->metric_reducers.reduce(
                /* index */ POSParser::KERNEL_in_memories,
                /* value */ 2
            );
            parser->metric_reducers.reduce(
                /* index */ POSParser::KERNEL_out_memories,
                /* value */ 1
            );
            parser->metric_counters.add_counter(
                /* index */ POSParser::KERNEL_number_of_vendor_kernels
            );
        #endif

    exit:
        return retval;
    }

} // namespace cublas_sgemm




/*!
 *  \related    cublasSgemmStridedBatched
 *  \brief      todo
 */
namespace cublas_sgemm_strided_batched {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandle_cuBLAS_Context *cublas_context_handle;
        POSHandle_CUDA_Memory *memory_handle_A, *memory_handle_B, *memory_handle_C;
        POSHandleManager_cuBLAS_Context *hm_cublas_context;
        POSHandleManager_CUDA_Memory *hm_memory;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 18)){
            POS_WARN(
                "parse(cublas_sgemm_strided_batched): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 18
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_cublas_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_cuBLAS_Context, POSHandleManager_cuBLAS_Context
        );
        POS_CHECK_POINTER(hm_cublas_context);

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // operate on handler manager
        retval = hm_cublas_context->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &cublas_context_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm_strided_batched): no cuBLAS context was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ cublas_context_handle
        });

        // operate on memory manager
        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 7, uint64_t),
            /* handle */ &memory_handle_A
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm_strided_batched): no memory handle A was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 7, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ memory_handle_A,
            /* param_index */ 7,
            /* offset */ pos_api_param_value(wqe, 7, uint64_t) - (uint64_t)(memory_handle_A->client_addr)
        });

        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 10, uint64_t),
            /* handle */ &memory_handle_B
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm_strided_batched): no memory handle B was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 10, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ memory_handle_B,
            /* param_index */ 10,
            /* offset */ pos_api_param_value(wqe, 10, uint64_t) - (uint64_t)(memory_handle_B->client_addr)
        });

        retval = hm_memory->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 14, uint64_t),
            /* handle */ &memory_handle_C
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cublas_sgemm_strided_batched): no memory handle C was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 14, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_Out>({
            /* handle */ memory_handle_C,
            /* param_index */ 14,
            /* offset */ pos_api_param_value(wqe, 14, uint64_t) - (uint64_t)(memory_handle_C->client_addr)
        });

        hm_memory->record_modified_handle(memory_handle_C);

        #if POS_CONF_RUNTIME_EnableTrace
            parser->metric_reducers.reduce(
                /* index */ POSParser::KERNEL_in_memories,
                /* value */ 2
            );
            parser->metric_reducers.reduce(
                /* index */ POSParser::KERNEL_out_memories,
                /* value */ 1
            );
            parser->metric_counters.add_counter(
                /* index */ POSParser::KERNEL_number_of_vendor_kernels
            );
        #endif

    exit:
        return retval;
    }

} // namespace cublas_sgemm_strided_batched




} // namespace ps_functions
