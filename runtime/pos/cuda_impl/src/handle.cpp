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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/cuda_impl/handle.h"


std::map<pos_resource_typeid_t, std::string> pos_resource_map = {
    {   kPOS_ResourceTypeId_Unknown,        "unknown"           },
    {   kPOS_ResourceTypeId_CUDA_Context,   "cuda_context",     },
    {   kPOS_ResourceTypeId_CUDA_Module,    "cuda_module",      },
    {   kPOS_ResourceTypeId_CUDA_Function,  "cuda_function",    },
    {   kPOS_ResourceTypeId_CUDA_Var,       "cuda_var",         },
    {   kPOS_ResourceTypeId_CUDA_Device,    "cuda_device",      },
    {   kPOS_ResourceTypeId_CUDA_Memory,    "cuda_memory",      },
    {   kPOS_ResourceTypeId_CUDA_Stream,    "cuda_stream",      },
    {   kPOS_ResourceTypeId_CUDA_Event,     "cuda_event",       },
    {   kPOS_ResourceTypeId_cuBLAS_Context, "cublas_context",   }
};


pos_retval_t POSHandle_CUDA::__sync_stream(uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;

    cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C(
            "failed to synchronize CUDA stream while processing handle: "
            "server_addr(%p), retval(%d)",
            this->server_addr, cuda_rt_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

exit:
    return retval;
}
