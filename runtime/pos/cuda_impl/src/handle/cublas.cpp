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

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/handle.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/handle/cublas.h"
#include "pos/cuda_impl/proto/cublas.pb.h"


POSHandle_cuBLAS_Context::POSHandle_cuBLAS_Context(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, uint64_t state_size)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size)
{
    this->resource_type_id = kPOS_ResourceTypeId_cuBLAS_Context;
}


POSHandle_cuBLAS_Context::POSHandle_cuBLAS_Context(size_t size_, void* hm, pos_u64id_t id_, uint64_t state_size) 
    : POSHandle_CUDA(size_, hm, id_, state_size)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


POSHandle_cuBLAS_Context::POSHandle_cuBLAS_Context(void* hm) 
    : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_cuBLAS_Context;
}


pos_retval_t POSHandle_cuBLAS_Context::tear_down(){
    pos_retval_t retval = POS_SUCCESS;
    cublasStatus_t cublas_retval;

    if(unlikely(this->status != kPOS_HandleStatus_Active)){ goto exit; }

    cublas_retval = cublasDestroy((cublasHandle_t)(this->server_addr));
    if(unlikely(cublas_retval != CUBLAS_STATUS_SUCCESS)){
        POS_WARN_C(
            "failed to tear down cuBLAS context: id(%lu), client_addr(%p), server_addr(%p)",
            this->id, this->client_addr, this->server_addr
        );
        retval = POS_FAILED;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_cuBLAS_Context::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_cuBLAS_Context::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_cuBLAS_Context *cublas_context_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cublas_context_binary = new pos_protobuf::Bin_POSHandle_cuBLAS_Context();
    POS_CHECK_POINTER(cublas_context_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cublas_context_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cublas_context_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_cuBLAS_Context::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    cublasHandle_t actual_handle;
    cublasStatus_t cublas_retval;
    POSHandle *stream_handle;

    POS_ASSERT(this->parent_handles.size() == 1);
    POS_CHECK_POINTER(stream_handle = this->parent_handles[0]);

    cublas_retval = cublasCreate_v2(&actual_handle);
    if(unlikely(CUBLAS_STATUS_SUCCESS != cublas_retval)){
        POS_WARN_C_DETAIL("failed to restore cublas context, failed to create: %d", cublas_retval);
        retval = POS_FAILED;
        goto exit;   
    }

    this->set_server_addr((void*)(actual_handle));
    this->mark_status(kPOS_HandleStatus_Active);

    cublas_retval = cublasSetStream(actual_handle, static_cast<cudaStream_t>(stream_handle->server_addr));
    if(unlikely(CUBLAS_STATUS_SUCCESS != cublas_retval)){
        POS_WARN_C_DETAIL("failed to restore cublas context, failed to pin to parent stream: %d", cublas_retval);
        retval = POS_FAILED;
        goto exit;   
    }

exit:
    return retval;
}


pos_retval_t POSHandleManager_cuBLAS_Context::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;

    this->_rid = kPOS_ResourceTypeId_cuBLAS_Context;

exit:
    return retval;
}


pos_retval_t POSHandleManager_cuBLAS_Context::allocate_mocked_resource(
    POSHandle_cuBLAS_Context** handle,
    std::map<uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *stream_handle;
    POS_CHECK_POINTER(handle);

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Stream) == 1);
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Stream].size() == 1);
    POS_CHECK_POINTER(stream_handle = related_handles[kPOS_ResourceTypeId_CUDA_Stream][0]);

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
    (*handle)->record_parent_handle(stream_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_cuBLAS_Context::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_cuBLAS_Context::try_restore_from_pool(POSHandle_cuBLAS_Context* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_cuBLAS_Context::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_cuBLAS_Context** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_cuBLAS_Context cublas_context_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cublas_context_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cublas_context_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cublas_context_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cublas_context_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cublas_context_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cublas_context_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cublas_context_binary.mutable_base()->id(),
        /* client_addr */ cublas_context_binary.mutable_base()->client_addr(),
        /* server_addr */ cublas_context_binary.mutable_base()->server_addr(),
        /* size */ cublas_context_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cublas_context_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cublas_context_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}
