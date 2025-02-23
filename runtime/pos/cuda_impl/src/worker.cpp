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
#include <algorithm>
#include <vector>
#include <map>
#include <thread>
#include <future>

#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/transport.h"
#include "pos/include/worker.h"
#include "pos/include/checkpoint.h"

#include "pos/cuda_impl/api_index.h"
#include "pos/cuda_impl/handle/memory.h"
#include "pos/cuda_impl/worker.h"


POSWorker_CUDA::POSWorker_CUDA(POSWorkspace* ws, POSClient* client)
    : POSWorker(ws, client) {}


POSWorker_CUDA::~POSWorker_CUDA(){}


pos_retval_t POSWorker_CUDA::sync(uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;

    cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C_DETAIL(
            "failed to synchronize worker, is this a bug?: stream_id(%p), cuda_rt_retval(%d)",
            stream_id, cuda_rt_retval
        );
        retval = POS_FAILED;
    }

    return retval;
}


pos_retval_t POSWorker_CUDA::daemon_init(){
    /*!
        *  \note   make sure the worker thread is bound to a CUDA context
        *          if we don't do this and use the driver API, it might be unintialized
        */
    if(cudaSetDevice(0) != cudaSuccess){
        POS_WARN_C_DETAIL("worker thread failed to invoke cudaSetDevice");
        return POS_FAILED; 
    }
    cudaDeviceSynchronize();
    
#if POS_CONF_EVAL_CkptOptLevel == 2
    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_ckpt_stream_id))
    );

    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_cow_stream_id))
    );
#endif

#if POS_CONF_EVAL_CkptOptLevel == 2 && POS_CONF_EVAL_CkptEnablePipeline == 1
    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_ckpt_commit_stream_id))
    );
#endif

#if POS_CONF_EVAL_MigrOptLevel == 2
    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_migration_precopy_stream_id))
    );
#endif

    return POS_SUCCESS; 
}


pos_retval_t POSWorker_CUDA::init_wk_functions() {
    this->_launch_functions.insert({
        /* CUDA runtime functions */
        {   CUDA_MALLOC,                    wk_functions::cuda_malloc::launch                       },
        {   CUDA_FREE,                      wk_functions::cuda_free::launch                         },
        {   CUDA_LAUNCH_KERNEL,             wk_functions::cuda_launch_kernel::launch                },
        {   CUDA_MEMCPY_HTOD,               wk_functions::cuda_memcpy_h2d::launch                   },
        {   CUDA_MEMCPY_DTOH,               wk_functions::cuda_memcpy_d2h::launch                   },
        {   CUDA_MEMCPY_DTOD,               wk_functions::cuda_memcpy_d2d::launch                   },
        {   CUDA_MEMCPY_HTOD_ASYNC,         wk_functions::cuda_memcpy_h2d_async::launch             },
        {   CUDA_MEMCPY_DTOH_ASYNC,         wk_functions::cuda_memcpy_d2h_async::launch             },
        {   CUDA_MEMCPY_DTOD_ASYNC,         wk_functions::cuda_memcpy_d2d_async::launch             },
        {   CUDA_MEMSET_ASYNC,              wk_functions::cuda_memset_async::launch                 },
        {   CUDA_SET_DEVICE,                wk_functions::cuda_set_device::launch                   },
        {   CUDA_GET_LAST_ERROR,            wk_functions::cuda_get_last_error::launch               },
        {   CUDA_GET_ERROR_STRING,          wk_functions::cuda_get_error_string::launch             },
        {   CUDA_PEEK_AT_LAST_ERROR,        wk_functions::cuda_peek_at_last_error::launch           },
        {   CUDA_GET_DEVICE_COUNT,          wk_functions::cuda_get_device_count::launch             },
        {   CUDA_GET_DEVICE_PROPERTIES,     wk_functions::cuda_get_device_properties::launch        },
        {   CUDA_DEVICE_GET_ATTRIBUTE,      wk_functions::cuda_device_get_attribute::launch         },
        {   CUDA_GET_DEVICE,                wk_functions::cuda_get_device::launch                   },
        {   CUDA_FUNC_GET_ATTRIBUTES,       wk_functions::cuda_func_get_attributes::launch          },
        {   CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS,   
                                    wk_functions::cuda_occupancy_max_active_bpm_with_flags::launch  },
        {   CUDA_STREAM_SYNCHRONIZE,        wk_functions::cuda_stream_synchronize::launch           },
        {   CUDA_STREAM_IS_CAPTURING,       wk_functions::cuda_stream_is_capturing::launch          },
        {   CUDA_EVENT_CREATE_WITH_FLAGS,   wk_functions::cuda_event_create_with_flags::launch      },
        {   CUDA_EVENT_DESTROY,             wk_functions::cuda_event_destory::launch                },
        {   CUDA_EVENT_RECORD,              wk_functions::cuda_event_record::launch                 },
        {   CUDA_EVENT_QUERY,               wk_functions::cuda_event_query::launch                  },
        
        /* CUDA driver functions */
        {   rpc_cuModuleLoad,               wk_functions::cu_module_load::launch                    },
        {   rpc_cuModuleLoadData,           wk_functions::cu_module_load_data::launch               },
        {   rpc_register_function,          wk_functions::__register_function::launch               },
        {   rpc_cuModuleGetFunction,        wk_functions::cu_module_get_function::launch            },
        {   rpc_register_var,               wk_functions::cu_module_get_global::launch              },
        {   rpc_cuDevicePrimaryCtxGetState, wk_functions::cu_device_primary_ctx_get_state::launch   },
        {   rpc_cuLaunchKernel,             wk_functions::cuda_launch_kernel::launch                },
        {   rpc_cuGetErrorString,           wk_functions::cu_get_error_string::launch               },
        
        /* cuBLAS functions */
        {   rpc_cublasCreate,               wk_functions::cublas_create::launch                     },
        {   rpc_cublasSetStream,            wk_functions::cublas_set_stream::launch                 },
        {   rpc_cublasSetMathMode,          wk_functions::cublas_set_math_mode::launch              },
        {   rpc_cublasSgemm,                wk_functions::cublas_sgemm::launch                      },
        {   rpc_cublasSgemmStridedBatched,  wk_functions::cublas_sgemm_strided_batched::launch      },
    });
    POS_DEBUG_C("insert %lu worker launch functions", this->_launch_functions.size());

    return POS_SUCCESS;
}


pos_retval_t POSWorker_CUDA::start_gpu_ticker(uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cudart_retval;
    cudaEvent_t start = (cudaEvent_t)(nullptr);

    if(unlikely(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0)){
        POS_WARN_C("start duplicated gpu ticker on the same CUDA stream, overwrite");
        cudart_retval = cudaEventDestroy(this->_cuda_ticker_events[(cudaStream_t)(stream_id)]);
        if(unlikely(cudart_retval != CUDA_SUCCESS)){
            POS_WARN_C("failed to destory old ticker CUDA event");
        }
    }

    cudart_retval = cudaEventCreate(&start);
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to create new ticker CUDA event");
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventRecord(start, (cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to start event record on specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    this->_cuda_ticker_events[(cudaStream_t)(stream_id)] = start;

exit:
    if(retval != POS_SUCCESS){
        if(start != (cudaEvent_t)(nullptr)){
            cudaEventDestroy(start);
        }
            
        if(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0){
            this->_cuda_ticker_events.erase((cudaStream_t)(stream_id));
        }
    }
    return retval;
}


pos_retval_t POSWorker_CUDA::stop_gpu_ticker(uint64_t& ticker, uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    float duration_ms = 0;
    cudaError_t cudart_retval;
    cudaEvent_t stop = (cudaEvent_t)(nullptr);

    if(unlikely(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) == 0)){
        POS_WARN_C("failed to stop gpu ticker, no start event exists");
        retval = POS_FAILED_NOT_EXIST;
        goto exit;
    }

    cudart_retval = cudaEventCreate(&stop);
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to create new ticker CUDA event");
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventRecord(stop, (cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to start event record on specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to sync specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventElapsedTime(
        &duration_ms, this->_cuda_ticker_events[(cudaStream_t)(stream_id)], stop
    );
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to elapsed time between CUDA events: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    POS_CHECK_POINTER(this->_ws);
    ticker = (uint64_t)(this->_ws->tsc_timer.ms_to_tick((uint64_t)(duration_ms)));

exit:
    if(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0){
        if(this->_cuda_ticker_events[(cudaStream_t)(stream_id)] != (cudaEvent_t)(nullptr)){
            cudaEventDestroy(this->_cuda_ticker_events[(cudaStream_t)(stream_id)]);
        }
        
        this->_cuda_ticker_events.erase((cudaStream_t)(stream_id));
    }
    if(stop != (cudaEvent_t)(nullptr)){
        cudaEventDestroy(stop);
    }

    return retval;
}
