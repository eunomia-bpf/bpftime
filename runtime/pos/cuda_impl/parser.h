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

#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/transport.h"
#include "pos/include/parser.h"

#include "pos/cuda_impl/api_index.h"

namespace ps_functions {
    /* CUDA runtime functions */
    POS_PS_DECLARE_FUNCTIONS(cuda_malloc);
    POS_PS_DECLARE_FUNCTIONS(cuda_free);
    POS_PS_DECLARE_FUNCTIONS(cuda_launch_kernel);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_h2d);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_d2h);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_d2d);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_h2d_async);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_d2h_async);
    POS_PS_DECLARE_FUNCTIONS(cuda_memcpy_d2d_async);
    POS_PS_DECLARE_FUNCTIONS(cuda_memset_async);
    POS_PS_DECLARE_FUNCTIONS(cuda_set_device);
    POS_PS_DECLARE_FUNCTIONS(cuda_get_last_error);
    POS_PS_DECLARE_FUNCTIONS(cuda_get_error_string);
    POS_PS_DECLARE_FUNCTIONS(cuda_peek_at_last_error);
    POS_PS_DECLARE_FUNCTIONS(cuda_get_device_count);
    POS_PS_DECLARE_FUNCTIONS(cuda_get_device_properties);
    POS_PS_DECLARE_FUNCTIONS(cuda_device_get_attribute);
    POS_PS_DECLARE_FUNCTIONS(cuda_get_device);
    POS_PS_DECLARE_FUNCTIONS(cuda_func_get_attributes);
    POS_PS_DECLARE_FUNCTIONS(cuda_occupancy_max_active_bpm_with_flags);
    POS_PS_DECLARE_FUNCTIONS(cuda_stream_synchronize);
    POS_PS_DECLARE_FUNCTIONS(cuda_stream_is_capturing);
    POS_PS_DECLARE_FUNCTIONS(cuda_event_create_with_flags);
    POS_PS_DECLARE_FUNCTIONS(cuda_event_destory);
    POS_PS_DECLARE_FUNCTIONS(cuda_event_record);
    POS_PS_DECLARE_FUNCTIONS(cuda_event_query);
    
    /* CUDA driver functions */
    POS_PS_DECLARE_FUNCTIONS(__register_function);   
    POS_PS_DECLARE_FUNCTIONS(cu_module_load); 
    POS_PS_DECLARE_FUNCTIONS(cu_module_load_data);    
    POS_PS_DECLARE_FUNCTIONS(cu_module_get_function);
    POS_PS_DECLARE_FUNCTIONS(cu_module_get_global);
    POS_PS_DECLARE_FUNCTIONS(cu_ctx_get_current);
    POS_PS_DECLARE_FUNCTIONS(cu_device_primary_ctx_get_state);
    POS_PS_DECLARE_FUNCTIONS(cu_get_error_string);

    /* cuBLAS functions */
    POS_PS_DECLARE_FUNCTIONS(cublas_create);
    POS_PS_DECLARE_FUNCTIONS(cublas_set_stream);
    POS_PS_DECLARE_FUNCTIONS(cublas_set_math_mode);
    POS_PS_DECLARE_FUNCTIONS(cublas_sgemm);
    POS_PS_DECLARE_FUNCTIONS(cublas_sgemm_strided_batched);

    /* remoting functions */
    POS_PS_DECLARE_FUNCTIONS(remoting_deinit);
} // namespace ps_functions

class POSClient_CUDA;

/*!
 *  \brief  POS Parser (CUDA Implementation)
 */
class POSParser_CUDA : public POSParser {
 public:
    POSParser_CUDA(POSWorkspace* ws, POSClient* client) : POSParser(ws, client){}
    ~POSParser_CUDA() = default;
    
 protected:
    /*!
     *  \brief      initialization of the runtime daemon thread
     *  \example    for CUDA, one need to call API e.g. cudaSetDevice first to setup the context for a thread
     */
    pos_retval_t daemon_init() override {
        return POS_SUCCESS; 
    }

    /*!
     *  \brief  insertion of parse functions
     *  \return POS_SUCCESS for succefully insertion
     */
    pos_retval_t init_ps_functions() override {
        this->_parser_functions.insert({
            /* CUDA runtime functions */
            {   CUDA_MALLOC,                    ps_functions::cuda_malloc::parse                        },
            {   CUDA_FREE,                      ps_functions::cuda_free::parse                          },
            {   CUDA_LAUNCH_KERNEL,             ps_functions::cuda_launch_kernel::parse                 },
            {   CUDA_MEMCPY_HTOD,               ps_functions::cuda_memcpy_h2d::parse                    },
            {   CUDA_MEMCPY_DTOH,               ps_functions::cuda_memcpy_d2h::parse                    },
            {   CUDA_MEMCPY_DTOD,               ps_functions::cuda_memcpy_d2d::parse                    },
            {   CUDA_MEMCPY_HTOD_ASYNC,         ps_functions::cuda_memcpy_h2d_async::parse              },
            {   CUDA_MEMCPY_DTOH_ASYNC,         ps_functions::cuda_memcpy_d2h_async::parse              },
            {   CUDA_MEMCPY_DTOD_ASYNC,         ps_functions::cuda_memcpy_d2d_async::parse              },
            {   CUDA_MEMSET_ASYNC,              ps_functions::cuda_memset_async::parse                  },
            {   CUDA_SET_DEVICE,                ps_functions::cuda_set_device::parse                    },
            {   CUDA_GET_LAST_ERROR,            ps_functions::cuda_get_last_error::parse                },
            {   CUDA_GET_ERROR_STRING,          ps_functions::cuda_get_error_string::parse              },
            {   CUDA_PEEK_AT_LAST_ERROR,        ps_functions::cuda_peek_at_last_error::parse            },
            {   CUDA_GET_DEVICE_COUNT,          ps_functions::cuda_get_device_count::parse              },
            {   CUDA_GET_DEVICE_PROPERTIES,     ps_functions::cuda_get_device_properties::parse         },
            {   CUDA_DEVICE_GET_ATTRIBUTE,      ps_functions::cuda_device_get_attribute::parse          },
            {   CUDA_GET_DEVICE,                ps_functions::cuda_get_device::parse                    },
            {   CUDA_FUNC_GET_ATTRIBUTES,       ps_functions::cuda_func_get_attributes::parse           },
            {   CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS,   
                                        ps_functions::cuda_occupancy_max_active_bpm_with_flags::parse   },
            {   CUDA_STREAM_SYNCHRONIZE,        ps_functions::cuda_stream_synchronize::parse            },
            {   CUDA_STREAM_IS_CAPTURING,       ps_functions::cuda_stream_is_capturing::parse           },
            {   CUDA_EVENT_CREATE_WITH_FLAGS,   ps_functions::cuda_event_create_with_flags::parse       },
            {   CUDA_EVENT_DESTROY,             ps_functions::cuda_event_destory::parse                 },
            {   CUDA_EVENT_RECORD,              ps_functions::cuda_event_record::parse                  },
            {   CUDA_EVENT_QUERY,               ps_functions::cuda_event_query::parse                   },

            /* CUDA driver functions */
            {   rpc_cuModuleLoad,               ps_functions::cu_module_load::parse                     },
            {   rpc_cuModuleLoadData,           ps_functions::cu_module_load_data::parse                },
            {   rpc_register_function,          ps_functions::__register_function::parse                },
            {   rpc_cuModuleGetFunction,        ps_functions::cu_module_get_function::parse             },
            {   rpc_register_var,               ps_functions::cu_module_get_global::parse               },
            {   rpc_cuCtxGetCurrent,            ps_functions::cu_ctx_get_current::parse                 },
            {   rpc_cuDevicePrimaryCtxGetState, ps_functions::cu_device_primary_ctx_get_state::parse    },
            {   rpc_cuLaunchKernel,             ps_functions::cuda_launch_kernel::parse                 },
            {   rpc_cuGetErrorString,           ps_functions::cu_get_error_string::parse                },
            
            /* cuBLAS functions */
            {   rpc_cublasCreate,               ps_functions::cublas_create::parse                      },
            {   rpc_cublasSetStream,            ps_functions::cublas_set_stream::parse                  },
            {   rpc_cublasSetMathMode,          ps_functions::cublas_set_math_mode::parse               },
            {   rpc_cublasSgemm,                ps_functions::cublas_sgemm::parse                       },
            {   rpc_cublasSgemmStridedBatched,  ps_functions::cublas_sgemm_strided_batched::parse       },

            /* remoting functgions */
            {   rpc_deinit,                     ps_functions::remoting_deinit::parse                    },
        });
        POS_DEBUG_C("insert %lu runtime parse functions", this->_parser_functions.size());

        return POS_SUCCESS;
    }
};
