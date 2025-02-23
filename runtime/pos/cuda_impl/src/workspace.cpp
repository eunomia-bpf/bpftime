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

#include "pos/cuda_impl/workspace.h"


POSWorkspace_CUDA::POSWorkspace_CUDA() : POSWorkspace(){}


pos_retval_t POSWorkspace_CUDA::__init(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult dr_retval;
    CUdevice cu_device;
    CUcontext cu_context;
    int device_count, i;

    // create the api manager
    this->api_mgnr = new POSApiManager_CUDA();
    POS_CHECK_POINTER(this->api_mgnr);
    this->api_mgnr->init();

    // mark all stateful resources
    this->resource_type_idx.insert(
        this->resource_type_idx.end(), {
            kPOS_ResourceTypeId_CUDA_Memory,
            kPOS_ResourceTypeId_CUDA_Context,
            kPOS_ResourceTypeId_CUDA_Module,
            kPOS_ResourceTypeId_CUDA_Function,
            kPOS_ResourceTypeId_CUDA_Var,
            kPOS_ResourceTypeId_CUDA_Device,
            kPOS_ResourceTypeId_CUDA_Stream,
            kPOS_ResourceTypeId_CUDA_Event,
            kPOS_ResourceTypeId_cuBLAS_Context
        }
    );
    this->stateful_resource_type_idx.insert(
        this->stateful_resource_type_idx.end(), {
            kPOS_ResourceTypeId_CUDA_Memory,
            kPOS_ResourceTypeId_CUDA_Module
        }
    );
    this->stateless_resource_type_idx.insert(
        this->stateless_resource_type_idx.end(), {
            kPOS_ResourceTypeId_CUDA_Context,
            kPOS_ResourceTypeId_CUDA_Function,
            kPOS_ResourceTypeId_CUDA_Var,
            kPOS_ResourceTypeId_CUDA_Device,
            kPOS_ResourceTypeId_CUDA_Stream,
            kPOS_ResourceTypeId_CUDA_Event,
            kPOS_ResourceTypeId_cuBLAS_Context
        }
    );

    dr_retval = cuInit(0);
    if(unlikely(dr_retval != CUDA_SUCCESS)){
        POS_ERROR_C_DETAIL("failed to initialize CUDA driver: dr_retval(%d)", dr_retval);
    }

    dr_retval = cuDeviceGetCount(&device_count);
    if (unlikely(dr_retval != CUDA_SUCCESS)) {
        POS_ERROR_C_DETAIL("failed to obtain number of CUDA devices: dr_retval(%d)", dr_retval);
    }
    if(unlikely(device_count <= 0)){
        POS_ERROR_C_DETAIL("no CUDA device detected on current machines");
    }

    // create one CUDA context on each device
    for(i=0; i<device_count; i++){
        // obtain handles on each device
        dr_retval = cuDeviceGet(&cu_device, i);
        if (unlikely(dr_retval != CUDA_SUCCESS)){
            POS_WARN_C("failed to obtain device handle of device %d, skipped: dr_retval(%d)", i, dr_retval);
            continue;
        }

        // create context
        dr_retval = cuCtxCreate(&cu_context, 0, cu_device);
        if (unlikely(dr_retval != CUDA_SUCCESS)) {
            POS_WARN_C("failed to create context on device %d, skipped: dr_retval(%d)", i, dr_retval);
            continue;
        }

        if(unlikely(i == 0)){
            // set the first device context as default context
            dr_retval = cuCtxSetCurrent(cu_context);
            if (dr_retval != CUDA_SUCCESS) {
                POS_WARN_C("failed to set context on device %d as current: dr_retval(%d)", i, dr_retval);
                retval = POS_FAILED_DRIVER;
                goto exit;
            }
        }
        this->_cu_contexts.push_back(cu_context);
        POS_DEBUG_C("created CUDA context: device_id(%d)", i);
    }

    if(unlikely(this->_cu_contexts.size() == 0)){
        POS_WARN_C("no CUDA context was created on any device");
        retval = POS_FAILED_DRIVER;
        goto exit;
    }

exit:
    if(unlikely(retval != POS_SUCCESS)){
        for(i=0; i<this->_cu_contexts.size(); i++){
            dr_retval = cuCtxDestroy(this->_cu_contexts[i]);
            if(unlikely(dr_retval != CUDA_SUCCESS)){
                POS_WARN_C(
                    "failed to destory context on device: device_id(%d), dr_retval(%d)",
                    i, dr_retval
                );
            } else {
                POS_DEBUG_C("destoried CUDA context: device_id(%d)", i);
            }
        }
        this->_cu_contexts.clear();
    }

    return retval;
}


pos_retval_t POSWorkspace_CUDA::__deinit(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult dr_retval;
    int i;

    for(i=0; i<this->_cu_contexts.size(); i++){
        POS_DEBUG("destorying cuda context on device...: device_id(%d)", i);
        dr_retval = cuCtxDestroy(this->_cu_contexts[i]);
        if(unlikely(dr_retval != CUDA_SUCCESS)){
            POS_WARN_C(
                "failed to destory context on device: device_id(%d), dr_retval(%d)",
                i, dr_retval
            );
        } else {
            POS_BACK_LINE
            POS_DEBUG_C("destoried cuda context on device: device_id(%d)", i);
        }
    }
    this->_cu_contexts.clear();

    return retval;
}


pos_retval_t POSWorkspace_CUDA::__create_client(pos_create_client_param_t& param, POSClient **client){
    pos_retval_t retval = POS_SUCCESS;
    pos_client_cxt_CUDA_t client_cxt;
    std::string runtime_daemon_log_path;
    std::string conf;

    POS_CHECK_POINTER(client);

    client_cxt.cxt_base.job_name = param.job_name;
    client_cxt.cxt_base.pid = param.pid;
    client_cxt.cxt_base.resource_type_idx = this->resource_type_idx;

    retval = this->ws_conf.get(POSWorkspaceConf::ConfigType::kRuntimeTraceResourceEnabled, conf);
    if(unlikely(retval != POS_SUCCESS)){
        POS_ERROR_C("failed to obtain resource trace mode in workspace configuration, this is a bug");
    }
    if(conf == "1"){ client_cxt.cxt_base.trace_resource = true; }
    else { client_cxt.cxt_base.trace_resource = false; }

    retval = this->ws_conf.get(POSWorkspaceConf::ConfigType::kRuntimeTracePerformanceEnabled, conf);
    if(unlikely(retval != POS_SUCCESS)){
        POS_ERROR_C("failed to obtain resource trace mode in workspace configuration, this is a bug");
    }
    if(conf == "1"){ client_cxt.cxt_base.trace_performance = true; }
    else { client_cxt.cxt_base.trace_performance = false; }

    retval = this->ws_conf.get(POSWorkspaceConf::ConfigType::kRuntimeDaemonLogPath, runtime_daemon_log_path);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to obtain runtime daemon log path");
        goto exit;
    } else {
        client_cxt.cxt_base.kernel_meta_path = runtime_daemon_log_path + std::string("/") 
                                                + param.job_name + std::string("_kernel_metas.txt");
    }

    POS_CHECK_POINTER(
        *client = new POSClient_CUDA(
            /* id */ param.id,
            /* pid */ param.pid,
            /* cxt */ client_cxt,
            /* ws */ this
        )
    );
    (*client)->init(param.is_restoring);

exit:
    return retval;
}


pos_retval_t POSWorkspace_CUDA::__destory_client(POSClient *client){
    pos_retval_t retval = POS_SUCCESS;
    POSClient_CUDA *cuda_client;

    POS_CHECK_POINTER(cuda_client = reinterpret_cast<POSClient_CUDA*>(client));
    cuda_client->deinit();

    /*!
     *  \note   we temp comment this out
     *  \todo   currently the client is data racing between OOB thread and RPC thread
     *          the RPC would have no idea the cuda_client has been deleted, so we don't delete tmply
     *          once we have the ability to spawn and recollect RPC thread from OOB callback
     *          we can resume this delete
     */
    // delete cuda_client;

exit:
    return retval;
}


pos_retval_t POSWorkspace_CUDA::preserve_resource(pos_resource_typeid_t rid, void *data){
    pos_retval_t retval = POS_SUCCESS;

    switch (rid)
    {
    case kPOS_ResourceTypeId_CUDA_Context:
        // no need to preserve context
        goto exit;
    
    case kPOS_ResourceTypeId_CUDA_Module:
        goto exit;

    default:
        retval = POS_FAILED_NOT_IMPLEMENTED;
        goto exit;
    }

exit:
    return retval;
}
