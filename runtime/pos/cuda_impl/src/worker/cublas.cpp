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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#include "pos/include/common.h"
#include "pos/cuda_impl/worker.h"


namespace wk_functions {


/*!
 *  \related    cuBlasCreate
 *  \brief      create a cuBlas context
 */
namespace cublas_create {
    // execution function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *cublas_context_handle;
        cublasHandle_t actual_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);
        
        // execute the actual cublasCreate
        wqe->api_cxt->return_code = cublasCreate_v2(&actual_handle);

        // record server address
        if(likely(CUBLAS_STATUS_SUCCESS == wqe->api_cxt->return_code)){
            cublas_context_handle = pos_api_create_handle(wqe, 0);
            POS_CHECK_POINTER(cublas_context_handle);
            cublas_context_handle->set_server_addr((void*)actual_handle);
            cublas_context_handle->mark_status(kPOS_HandleStatus_Active);
        }

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

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
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *stream_handle, *cublas_context_handle, *new_parent_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        cublas_context_handle = pos_api_input_handle(wqe, 1);
        POS_CHECK_POINTER(cublas_context_handle);

        wqe->api_cxt->return_code = cublasSetStream(
            (cublasHandle_t)(cublas_context_handle->server_addr),
            (cudaStream_t)(stream_handle->server_addr)
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
            goto exit;
        }

        // confirm parent change
        POS_ASSERT(cublas_context_handle->parent_handles.size() == 1);
        POS_CHECK_POINTER(cublas_context_handle->parent_handles[0] = stream_handle);

        POSWorker::__done(ws, wqe);

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
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *cublas_context_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        cublas_context_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(cublas_context_handle);

        wqe->api_cxt->return_code = cublasSetMathMode(
            (cublasHandle_t)(cublas_context_handle->server_addr),
            pos_api_param_value(wqe, 1, cublasMath_t)
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

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
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *cublas_context_handle;
        POSHandle *memory_handle_A, *memory_handle_B, *memory_handle_C;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        cublas_context_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(cublas_context_handle);

        memory_handle_A = pos_api_input_handle(wqe, 1);
        POS_CHECK_POINTER(memory_handle_A);
        memory_handle_B = pos_api_input_handle(wqe, 2);
        POS_CHECK_POINTER(memory_handle_B);
        memory_handle_C = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle_C);

        wqe->api_cxt->return_code = cublasSgemm(
            /* handle */ (cublasHandle_t)(cublas_context_handle->server_addr),
            /* transa */ pos_api_param_value(wqe, 1, cublasOperation_t),
            /* transb */ pos_api_param_value(wqe, 2, cublasOperation_t),
            /* m */ pos_api_param_value(wqe, 3, int),
            /* n */ pos_api_param_value(wqe, 4, int),
            /* k */ pos_api_param_value(wqe, 5, int),
            /* alpha */ (float*)pos_api_param_addr(wqe, 6),
            /* A */ (float*)(pos_api_input_handle_offset_server_addr(wqe, 1)),
            /* lda */ pos_api_param_value(wqe, 8, int),
            /* B */ (float*)(pos_api_input_handle_offset_server_addr(wqe, 2)),
            /* ldb */ pos_api_param_value(wqe, 10, int),
            /* beta */ (float*)pos_api_param_addr(wqe, 11),
            /* C */ (float*)(pos_api_output_handle_offset_server_addr(wqe, 0)),
            /* ldc */ pos_api_param_value(wqe, 13, int)
        );

        if(unlikely(CUBLAS_STATUS_SUCCESS != wqe->api_cxt->return_code)){
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

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
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *cublas_context_handle;
        POSHandle *memory_handle_A, *memory_handle_B, *memory_handle_C;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        cublas_context_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(cublas_context_handle);

        memory_handle_A = pos_api_input_handle(wqe, 1);
        POS_CHECK_POINTER(memory_handle_A);
        memory_handle_B = pos_api_input_handle(wqe, 2);
        POS_CHECK_POINTER(memory_handle_B);
        memory_handle_C = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle_C);

        wqe->api_cxt->return_code = cublasSgemmStridedBatched(
            /* handle */ (cublasHandle_t)(cublas_context_handle->server_addr),
            /* transa */ pos_api_param_value(wqe, 1, cublasOperation_t),
            /* transb */ pos_api_param_value(wqe, 2, cublasOperation_t),
            /* m */ pos_api_param_value(wqe, 3, int),
            /* n */ pos_api_param_value(wqe, 4, int),
            /* k */ pos_api_param_value(wqe, 5, int),
            /* alpha */ (float*)pos_api_param_addr(wqe, 6),
            /* A */ (float*)(pos_api_input_handle_offset_server_addr(wqe, 1)),
            /* lda */ pos_api_param_value(wqe, 8, int),
            /* sA */ pos_api_param_value(wqe, 9, long long int),
            /* B */ (float*)(pos_api_input_handle_offset_server_addr(wqe, 2)),
            /* ldb */ pos_api_param_value(wqe, 11, int),
            /* sB */ pos_api_param_value(wqe, 12, long long int),
            /* beta */ (float*)pos_api_param_addr(wqe, 13),
            /* C */ (float*)(pos_api_output_handle_offset_server_addr(wqe, 0)),
            /* ldc */ pos_api_param_value(wqe, 15, int),
            /* sC */ pos_api_param_value(wqe, 16, long long int),
            /* batchCount */ pos_api_param_value(wqe, 17, int)
        );

        if(unlikely(CUBLAS_STATUS_SUCCESS != wqe->api_cxt->return_code)){
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cublas_sgemm_strided_batched


} // namespace wk_functions
