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
#include "pos/cuda_impl/handle/stream.h"
#include "pos/cuda_impl/proto/stream.pb.h"


POSHandle_CUDA_Stream::POSHandle_CUDA_Stream(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Stream;
}


POSHandle_CUDA_Stream::POSHandle_CUDA_Stream(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Stream;
}


POSHandle_CUDA_Stream::POSHandle_CUDA_Stream(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Stream::tear_down(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;
    CUcontext pctx;

    if(unlikely(this->status != kPOS_HandleStatus_Active)){ goto exit; }

    cuda_dv_retval = cuCtxGetCurrent(&pctx);
    if(unlikely(cuda_dv_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to get cu context: retval(%d)", cuda_dv_retval);
    }

    cuda_dv_retval = cuStreamSynchronize((CUstream)(this->server_addr));
    if(unlikely(cuda_dv_retval != CUDA_SUCCESS)){
        POS_WARN_C(
            "failed to tear down CUDA stream, failed to call cuStreamSynchronize: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

    cuda_dv_retval = cuStreamDestroy((CUstream)(this->server_addr));
    if(unlikely(cuda_dv_retval != CUDA_SUCCESS)){
        POS_WARN_C(
            "failed to tear down CUDA stream, failed to call cuStreamDestroy: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Stream::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Stream::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Stream *cuda_stream_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_stream_binary = new pos_protobuf::Bin_POSHandle_CUDA_Stream();
    POS_CHECK_POINTER(cuda_stream_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_stream_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_stream_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_CUDA_Stream::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_res;
    CUstream stream_addr;

    CUcontext pctx;
    cuda_dv_res = cuCtxGetCurrent(&pctx);
    if(unlikely(cuda_dv_res != CUDA_SUCCESS)){
        POS_WARN_C("failed to get cu context: retval(%d)", cuda_dv_res);
    }

    if((cuda_dv_res = cuStreamCreate(&stream_addr, CU_STREAM_DEFAULT)) != CUDA_SUCCESS){
        POS_WARN_C("cuStreamCreate failed: %d", cuda_dv_res);
        retval = POS_FAILED_DRIVER;
        goto exit;
    }
    this->set_server_addr((void*)(stream_addr));
    this->mark_status(kPOS_HandleStatus_Active);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Stream::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle_CUDA_Stream *stream_handle;

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Context) == 1);

    this->_rid = kPOS_ResourceTypeId_CUDA_Stream;

    /*!
     *  \note   we won't use the default stream, and we will create a new non-default stream
     *          within the worker thread, so that we can achieve overlap checkpointing
     *  \todo   we bind this default stream to all upstream context, actually this isn't 
     *          correct, we need to creat one default stream for each context
     *          we would implement this once we have context system
     */
    if(unlikely(POS_SUCCESS != this->allocate_mocked_resource(
        /* handle */ &stream_handle,
        /* related_handle */ std::map<uint64_t, std::vector<POSHandle*>>({
            { kPOS_ResourceTypeId_CUDA_Context, related_handles[kPOS_ResourceTypeId_CUDA_Context] }
        }),
        /* size */ sizeof(cudaStream_t),
        /* use_expected_addr */ true,
        /* expected_addr */ 0
    ))){
        POS_ERROR_C_DETAIL("failed to allocate mocked CUDA stream in the manager");
    }

    // record in the manager
    this->latest_used_handle = this->_handles[0];
    this->default_handle = this->_handles[0];

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Stream::allocate_mocked_resource(
    POSHandle_CUDA_Stream** handle,
    std::map</* type */ uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *context_handle;
    uint64_t i;

    POS_CHECK_POINTER(handle);

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Context) == 1);

    // TODO: set to == 1 when we have context control
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Context].size() >= 1);

    POS_CHECK_POINTER(context_handle = related_handles[kPOS_ResourceTypeId_CUDA_Context][0]);

    retval = this->__allocate_mocked_resource(
        /* handle */ handle,
        /* size */ size,
        /* use_expected_addr */ use_expected_addr,
        /* expected_addr */ expected_addr,
        /* state_size */ state_size
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA stream in the manager");
        goto exit;
    }

    POS_CHECK_POINTER(*handle);

    // TODO: set to "(*handle)->record_parent_handle(context_handle);" when we have context control
    for(i=0; i<related_handles[kPOS_ResourceTypeId_CUDA_Context].size(); i++){
        POS_CHECK_POINTER(context_handle = related_handles[kPOS_ResourceTypeId_CUDA_Context][i]);
        (*handle)->record_parent_handle(context_handle);
    }

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Stream::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Stream::try_restore_from_pool(POSHandle_CUDA_Stream* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Stream::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Stream** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Stream cuda_stream_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_stream_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_stream_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_stream_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_stream_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_stream_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_stream_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_stream_binary.mutable_base()->id(),
        /* client_addr */ cuda_stream_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_stream_binary.mutable_base()->server_addr(),
        /* size */ cuda_stream_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_stream_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_stream_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}
