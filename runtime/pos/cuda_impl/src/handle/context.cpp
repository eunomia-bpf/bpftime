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
#include <string>
#include <cstdlib>

#include <sys/resource.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/handle.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/handle/context.h"
#include "pos/cuda_impl/proto/context.pb.h"


POSHandle_CUDA_Context::POSHandle_CUDA_Context(
    void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_
)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Context;
}


POSHandle_CUDA_Context::POSHandle_CUDA_Context(void* hm) : POSHandle_CUDA(hm){
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Context;
}


POSHandle_CUDA_Context::POSHandle_CUDA_Context(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}



pos_retval_t POSHandle_CUDA_Context::tear_down(){
    return POS_FAILED_NOT_IMPLEMENTED;
}


pos_retval_t POSHandle_CUDA_Context::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Context::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Context *cuda_context_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_context_binary = new pos_protobuf::Bin_POSHandle_CUDA_Context();
    POS_CHECK_POINTER(cuda_context_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_context_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_context_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_CUDA_Context::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_res;
    CUresult cuda_dv_res;
    CUcontext pctx;
    POSHandle *parent_device_handle;
    int origin_device_id;

    // obtain device id to resume after restore this context
    // shall we find a more elegant way to do context switch?
    if(cuda_rt_res = cudaGetDevice(&origin_device_id)){
        retval = POS_FAILED_DRIVER;
        POS_WARN_C_DETAIL("failed to restore CUDA context, failed to obtain device id: %d", cuda_rt_res);
        goto exit;
    }

    POS_ASSERT(this->parent_handles.size() == 1);
    POS_CHECK_POINTER(parent_device_handle = this->parent_handles[0]);
    POS_ASSERT(parent_device_handle->resource_type_id == kPOS_ResourceTypeId_CUDA_Device);

    if(unlikely(cudaSuccess != (
        cuda_rt_res = cudaSetDevice(static_cast<int>((uint64_t)(parent_device_handle->client_addr)))
    ))){
        retval = POS_FAILED_DRIVER;
        POS_WARN_C_DETAIL("failed to restore CUDA context, cudaSetDevice failed: %d", cuda_rt_res);
        goto exit;
    }
    cudaDeviceSynchronize();

    // obtain current cuda context
    if((cuda_dv_res = cuCtxGetCurrent(&pctx)) != CUDA_SUCCESS){
        retval = POS_FAILED;
        POS_WARN_C_DETAIL("failed to restore CUDA context, cuCtxGetCurrent failed: %d", cuda_dv_res);
        goto exit;
    }
    this->set_server_addr((void*)pctx);
    this->mark_status(kPOS_HandleStatus_Active);

    // switch back
    if((cuda_rt_res = cudaSetDevice(origin_device_id)) != cudaSuccess){
        retval = POS_FAILED_DRIVER;
        POS_WARN_C_DETAIL("restore CUDA context, but failed to switch back to old device, cudaSetDevice failed: %d", cuda_rt_res);
        goto exit;
    }

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Context::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i, nb_device;
    POSHandle *device_handle;
    POSHandle_CUDA_Context *ctx_handle;
    
    this->_rid = kPOS_ResourceTypeId_CUDA_Context;

    if(unlikely(related_handles.count(kPOS_ResourceTypeId_CUDA_Device) == 0)){
        retval = POS_FAILED_INVALID_INPUT;
        POS_WARN_C("failed to init handle manager for CUDA context, no device provided");
        goto exit;
    }

    nb_device = related_handles[kPOS_ResourceTypeId_CUDA_Device].size();
    if(unlikely(nb_device == 0)){
        retval = POS_FAILED_INVALID_INPUT;
        POS_WARN_C("failed to init handle manager for CUDA context, no device provided");
        goto exit;
    }

    for(i=0; i<nb_device; i++){
        POS_CHECK_POINTER(device_handle = related_handles[kPOS_ResourceTypeId_CUDA_Device][i]);
    
        // allocate mocked context, and setup the actual context address
        if(unlikely(POS_SUCCESS != (
            retval = this->allocate_mocked_resource(
                /* handle */ &ctx_handle,
                /* related_handle */ std::map<uint64_t, std::vector<POSHandle*>>({
                    { kPOS_ResourceTypeId_CUDA_Device, { device_handle } }
                }),
                /* size */ 1,
                /* use_expected_addr */ true,
                /* expected_addr */ static_cast<uint64_t>(i),   // device id == context id
                /* state_size */ 0
            )
        ))){
            POS_WARN_C_DETAIL(
                "failed to allocate mocked CUDA context in the manager: device_id(%d)",
                static_cast<int>((uint64_t)(device_handle->client_addr))
            );
            continue;
        }        
    }
    this->latest_used_handle = this->_handles[0];

    // here we need to bind this context to real device context,
    // which we have alreay created by workspace
    for(i=0; i<this->_handles.size(); i++){
        POS_CHECK_POINTER(ctx_handle = this->get_handle_by_id(i));
        if(unlikely(POS_SUCCESS != (
            retval = ctx_handle->__restore()
        ))){
            POS_WARN_C("failed to bind CUDA context to real device: device_id(%d)", i);
            goto exit;
        }
    }

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Context::allocate_mocked_resource(
    POSHandle_CUDA_Context** handle,
    std::map<uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *device_handle;

    POS_CHECK_POINTER(handle);

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Device) == 1);
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Device].size() == 1);
    POS_CHECK_POINTER(device_handle = related_handles[kPOS_ResourceTypeId_CUDA_Device][0]);

    retval = this->__allocate_mocked_resource(
        /* handle */ handle,
        /* size */ size,
        /* use_expected_addr */ use_expected_addr,
        /* expected_addr */ expected_addr,
        /* state_size */ state_size
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA module in the manager");
        goto exit;
    }

    POS_CHECK_POINTER(*handle);
    (*handle)->record_parent_handle(device_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Context::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Context::try_restore_from_pool(POSHandle_CUDA_Context* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Context::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Context** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Context cuda_context_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_context_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_context_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_context_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_context_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_context_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_context_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_context_binary.mutable_base()->id(),
        /* client_addr */ cuda_context_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_context_binary.mutable_base()->server_addr(),
        /* size */ cuda_context_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_context_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_context_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}
