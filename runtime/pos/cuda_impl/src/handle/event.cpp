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
#include "pos/cuda_impl/handle/event.h"
#include "pos/cuda_impl/proto/event.pb.h"


POSHandle_CUDA_Event::POSHandle_CUDA_Event(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Event;
}


POSHandle_CUDA_Event::POSHandle_CUDA_Event(void* hm) : POSHandle_CUDA(hm) {
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Event;
}


POSHandle_CUDA_Event::POSHandle_CUDA_Event(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_) 
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Event::tear_down(){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;

    if(unlikely(this->status != kPOS_HandleStatus_Active)){ goto exit; }

    cuda_rt_retval = cudaEventDestroy((cudaEvent_t)(this->server_addr));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C(
            "failed to tear down CUDA event: id(%lu), client_addr(%p), server_addr(%p)",
            this->id, this->client_addr, this->server_addr
        );
        retval = POS_FAILED;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Event::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Event::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Event *cuda_event_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_event_binary = new pos_protobuf::Bin_POSHandle_CUDA_Event();
    POS_CHECK_POINTER(cuda_event_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_event_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_event_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    cuda_event_binary->set_flags(this->flags);

    return retval;
}


pos_retval_t POSHandle_CUDA_Event::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_res;
    cudaEvent_t ptr;

    cuda_rt_res = cudaEventCreateWithFlags(&ptr, this->flags);
    if(unlikely(cuda_rt_res != cudaSuccess)){
        POS_WARN_C_DETAIL("failed to restore CUDA event, create failed: retval(%d)", cuda_rt_res);
        retval = POS_FAILED;
        goto exit;
    }

    this->set_server_addr((void*)(ptr));
    this->mark_status(kPOS_HandleStatus_Active);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Event::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;

    this->_rid = kPOS_ResourceTypeId_CUDA_Event;

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Event::allocate_mocked_resource(
    POSHandle_CUDA_Event** handle,
    std::map<uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *context_handle;
    POS_CHECK_POINTER(handle);

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Context) == 1);
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Context].size() == 1);
    POS_CHECK_POINTER(context_handle = related_handles[kPOS_ResourceTypeId_CUDA_Context][0]);

    retval = this->__allocate_mocked_resource(
        /* handle */ handle,
        /* size */ size,
        /* use_expected_addr */ use_expected_addr,
        /* expected_addr */ expected_addr,
        /* state_size */ state_size
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked cuBLAS context in the manager");
        goto exit;
    }

    POS_CHECK_POINTER(*handle);
    (*handle)->record_parent_handle(context_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Event::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Event::try_restore_from_pool(POSHandle_CUDA_Event* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Event::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Event** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Event cuda_event_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_event_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_event_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_event_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_event_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_event_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_event_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_event_binary.mutable_base()->id(),
        /* client_addr */ cuda_event_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_event_binary.mutable_base()->server_addr(),
        /* size */ cuda_event_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_event_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_event_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}
