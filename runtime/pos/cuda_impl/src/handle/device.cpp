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
#include "pos/cuda_impl/handle/device.h"
#include "pos/cuda_impl/proto/device.pb.h"


POSHandle_CUDA_Device::POSHandle_CUDA_Device(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Device;
}


POSHandle_CUDA_Device::POSHandle_CUDA_Device(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Device;
}


POSHandle_CUDA_Device::POSHandle_CUDA_Device(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Device::tear_down(){
    return POS_FAILED_NOT_IMPLEMENTED;
}


pos_retval_t POSHandle_CUDA_Device::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Device::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Device *cuda_device_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_device_binary = new pos_protobuf::Bin_POSHandle_CUDA_Device();
    POS_CHECK_POINTER(cuda_device_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_device_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_device_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_CUDA_Device::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;
    cudaDeviceProp prop;

    // invoke cudaGetDeviceProperties here to make sure the device is alright
    cuda_rt_retval = cudaGetDeviceProperties(&prop, this->id);
    
    if(unlikely(cuda_rt_retval == cudaSuccess)){
        this->mark_status(kPOS_HandleStatus_Active);
    } else {
        POS_WARN_C_DETAIL("failed to restore CUDA device, cudaGetDeviceProperties failed: %d, device_id(%d)", cuda_rt_retval, this->id);
        retval = POS_FAILED;
    } 

    return retval;
}


pos_retval_t POSHandleManager_CUDA_Device::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    int num_device, i;
    cudaError_t cuda_rt_retval;
    POSHandle_CUDA_Device *device_handle;

    POS_ASSERT(related_handles.size() == 0);

    this->_rid = kPOS_ResourceTypeId_CUDA_Device;

    // get number of physical devices on the machine
    if(unlikely(cudaSuccess != (
        cuda_rt_retval = cudaGetDeviceCount(&num_device)
    ))){
        POS_WARN_C("failed to call cudaGetDeviceCount: retval(%d)", cuda_rt_retval);
        retval = POS_FAILED_DRIVER;
        goto exit;
    }
    if(unlikely(num_device == 0)){ 
        POS_WARN_C("no cuda device, POS won't be enabled");
        retval = POS_FAILED_DRIVER;
        goto exit;
    }

    for(i=0; i<num_device; i++){
        if(unlikely(
            POS_SUCCESS != this->allocate_mocked_resource(
                /* handle */ &device_handle,
                /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({}),
                /* size */ 1,
                /* use_expected_addr */ true,
                /* expected_addr */ static_cast<uint64_t>(i),
                /* state_size */ 0
            )
        )){
            POS_ERROR_C_DETAIL("failed to allocate mocked CUDA device in the manager");
        }
        device_handle->mark_status(kPOS_HandleStatus_Active);
    }
    this->latest_used_handle = this->_handles[0];

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Device::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Device::try_restore_from_pool(POSHandle_CUDA_Device* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Device::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Device** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Device device_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!device_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(device_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = device_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = device_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = device_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = device_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ device_binary.mutable_base()->id(),
        /* client_addr */ device_binary.mutable_base()->client_addr(),
        /* server_addr */ device_binary.mutable_base()->server_addr(),
        /* size */ device_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ device_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            device_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}
