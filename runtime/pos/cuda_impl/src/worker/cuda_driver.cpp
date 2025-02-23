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
#include "pos/cuda_impl/worker.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

char* __cu_demangle(const char *id, char *output_buffer, size_t *length, int *status);

namespace wk_functions {

/*!
 *  \related    cuModuleLoadData
 *  \brief      load CUmodule down to the driver, which contains PTX/SASS binary
 */
namespace cu_module_load {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Module *module_handle;
        CUresult res;
        CUmodule module = nullptr, patched_module = nullptr;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        module_handle = reinterpret_cast<POSHandle_CUDA_Module*>(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(module_handle);

        // create normal module
        wqe->api_cxt->return_code = cuModuleLoadData(
            /* module */ &module,
            /* image */  pos_api_param_addr(wqe, 1)
        );
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            module_handle->set_server_addr((void*)module);
            module_handle->mark_status(kPOS_HandleStatus_Active); // TODO: remove this
        } else {
            POS_WARN("failed to cuModuleLoadData normal module")
        }

        // create patched module
        // wqe->api_cxt->return_code = cuModuleLoadData(
        //     /* module */ &patched_module,
        //     /* image */ (void*)(module_handle->patched_binary.data())
        // );
        // if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
        //     module_handle->patched_server_addr = (void*)(patched_module);
        //     module_handle->mark_status(kPOS_HandleStatus_Active);
        // } else {
        //     POS_WARN("failed to cuModuleLoadData patched module")
        // }

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cu_module_load



/*!
 *  \related    cuModuleLoadData
 *  \brief      load CUmodule down to the driver, which contains PTX/SASS binary
 */
namespace cu_module_load_data {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Module *module_handle;
        CUresult res;
        CUmodule module = nullptr, patched_module = nullptr;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        module_handle = reinterpret_cast<POSHandle_CUDA_Module*>(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(module_handle);

        // create normal module
        wqe->api_cxt->return_code = cuModuleLoadData(
            /* module */ &module,
            /* image */  pos_api_param_addr(wqe, 0)
        );
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            module_handle->set_server_addr((void*)module);
            module_handle->mark_status(kPOS_HandleStatus_Active); // TODO: remove this
        } else {
            POS_WARN("failed to cuModuleLoadData normal module")
        }

        // create patched module
        // wqe->api_cxt->return_code = cuModuleLoadData(
        //     /* module */ &patched_module,
        //     /* image */ (void*)(module_handle->patched_binary.data())
        // );
        // if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
        //     module_handle->patched_server_addr = (void*)(patched_module);
        //     module_handle->mark_status(kPOS_HandleStatus_Active);
        // } else {
        //     POS_WARN("failed to cuModuleLoadData patched module")
        // }

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cu_module_load_data




/*!
 *  \related    __cudaRegisterFunction 
 *  \brief      implicitly register cuda function
 */
namespace __register_function {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *module_handle;
        POSHandle_CUDA_Function *function_handle;
        CUfunction function = NULL;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);
    
        function_handle = (POSHandle_CUDA_Function*)(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

        POS_ASSERT(function_handle->parent_handles.size() > 0);
        module_handle = function_handle->parent_handles[0];

        wqe->api_cxt->return_code = cuModuleGetFunction(
            &function, (CUmodule)(module_handle->server_addr), function_handle->name.c_str()
        );

        // record server address
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            function_handle->set_server_addr((void*)function);
            function_handle->mark_status(kPOS_HandleStatus_Active);
        }

        // TODO: skip checking
        // if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
        //     POSWorker::__restore(ws, wqe);
        // } else {
        //     POSWorker::__done(ws, wqe);
        // }
        POSWorker::__done(ws, wqe);

    exit:
        return retval;
    }
} // namespace __register_function




/*!
 *  \related    cuModuleGetFunction 
 *  \brief      obtain kernel host pointer by given kernel name from specified CUmodule
 */
namespace cu_module_get_function {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *module_handle;
        POSHandle_CUDA_Function *function_handle;
        CUfunction function = NULL;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);
    
        function_handle = (POSHandle_CUDA_Function*)(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

        POS_ASSERT(function_handle->parent_handles.size() > 0);
        module_handle = function_handle->parent_handles[0];

        wqe->api_cxt->return_code = cuModuleGetFunction(
            &function, (CUmodule)(module_handle->server_addr), function_handle->name.c_str()
        );

        // record server address
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            function_handle->set_server_addr((void*)function);
            function_handle->mark_status(kPOS_HandleStatus_Active);
        }

        // TODO: skip checking
        // if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
        //     POSWorker::__restore(ws, wqe);
        // } else {
        //     POSWorker::__done(ws, wqe);
        // }
        POSWorker::__done(ws, wqe);

    exit:
        return retval;
    }
} // namespace cu_module_get_function


/*!
 *  \related    cuModuleGetGlobal
 *  \brief      obtain the host-side pointer of a global CUDA symbol
 */
namespace cu_module_get_global {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *module_handle;
        POSHandle_CUDA_Var *var_handle;
        CUfunction function = NULL;

        CUdeviceptr dptr = 0;
        size_t d_size = 0;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        var_handle = (POSHandle_CUDA_Var*)(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(var_handle);

        POS_ASSERT(var_handle->parent_handles.size() > 0);
        module_handle = var_handle->parent_handles[0];

        wqe->api_cxt->return_code = cuModuleGetGlobal(
            &dptr, &d_size, (CUmodule)(module_handle->server_addr), var_handle->global_name.c_str()
        );

        // record server address
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            var_handle->set_server_addr((void*)dptr);
            var_handle->mark_status(kPOS_HandleStatus_Active);
        }

        // we temp hide the error from this api
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            wqe->api_cxt->return_code = CUDA_SUCCESS;
        }

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cu_module_get_global




/*!
 *  \related    cuDevicePrimaryCtxGetState
 *  \brief      obtain the state of the primary context
 */
namespace cu_device_primary_ctx_get_state {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Device *device_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        wqe->api_cxt->return_code = cuDevicePrimaryCtxGetState(
            device_handle->id,
            (unsigned int*)(wqe->api_cxt->ret_data),
            (int*)(wqe->api_cxt->ret_data + sizeof(unsigned int))
        );

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cu_device_primary_ctx_get_state


/*!
 *  \related    cuCtxGetCurrent
 *  \brief      obtain the state of the current context
 */
namespace cu_ctx_get_current {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        POS_ERROR_DETAIL("shouldn't be called");
        return POS_SUCCESS;
    }
} // namespace cu_ctx_get_current


/*!
 *  \related    cuGetErrorString
 *  \brief      obtain the error string from the CUDA context
 */
namespace cu_get_error_string {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        const char* ret_string;
        wqe->api_cxt->return_code = cuGetErrorString(pos_api_param_value(wqe, 0, CUresult), &ret_string);

        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            if(likely(strlen(ret_string) > 0)){
                POS_ASSERT(strlen(ret_string)+1 < 128);
                memcpy(wqe->api_cxt->ret_data, ret_string, strlen(ret_string)+1);
            }
        }

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }
        
        return POS_SUCCESS;
    }
} // namespace cu_get_error_string


} // namespace wk_functions 
