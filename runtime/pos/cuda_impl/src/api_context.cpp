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
#include <vector>

#include <stdint.h>


#include "pos/include/common.h"
#include "pos/include/api_context.h"
#include "pos/cuda_impl/api_index.h"

/*!
 *  \brief  all POS-hijacked should be record here, used for debug checking
 */
std::vector<uint64_t> pos_hijacked_cuda_apis({
    /* CUDA Runtime */
    CUDA_MALLOC,
    CUDA_FREE,
    CUDA_LAUNCH_KERNEL,
    CUDA_MEMCPY_HTOD,
    CUDA_MEMCPY_DTOH,
    CUDA_MEMCPY_DTOD,
    CUDA_MEMCPY_HTOD_ASYNC,
    CUDA_MEMCPY_DTOH_ASYNC,
    CUDA_MEMCPY_DTOD_ASYNC,
    CUDA_MEMSET_ASYNC,
    CUDA_SET_DEVICE,
    CUDA_GET_LAST_ERROR,
    CUDA_GET_ERROR_STRING,
    CUDA_PEEK_AT_LAST_ERROR,
    CUDA_GET_DEVICE_COUNT,
    CUDA_GET_DEVICE_PROPERTIES,
    CUDA_DEVICE_GET_ATTRIBUTE,
    CUDA_GET_DEVICE,
    CUDA_FUNC_GET_ATTRIBUTES,
    CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS,
    CUDA_STREAM_SYNCHRONIZE,
    CUDA_STREAM_IS_CAPTURING,
    CUDA_EVENT_CREATE_WITH_FLAGS,
    CUDA_EVENT_DESTROY,
    CUDA_EVENT_RECORD,
    CUDA_EVENT_QUERY,

    /* CUDA Driver */
    rpc_cuModuleLoad,
    rpc_cuModuleLoadData,
    rpc_cuModuleGetFunction,
    rpc_register_function,
    rpc_register_var,
    rpc_cuDevicePrimaryCtxGetState,
    rpc_cuCtxGetCurrent,
    rpc_cuLaunchKernel,

    /* cuBLAS */
    rpc_cublasCreate,
    rpc_cublasSetStream,
    rpc_cublasSetMathMode,
    rpc_cublasSgemm,
    rpc_cublasSgemmStridedBatched,
    rpc_cuGetErrorString,

    /* remoting */
    rpc_deinit,

    /* no need to hijack */
    rpc_printmessage,
    rpc_elf_load,
    rpc_register_function,
    rpc_elf_unload,
    rpc_checkpoint
});

bool pos_is_hijacked(uint64_t api_id){
    uint64_t i=0;
    for(i=0; i<pos_hijacked_cuda_apis.size(); i++){
        if(unlikely(pos_hijacked_cuda_apis[i] == api_id)){
            return true;
        }
    }
    return false;
}
