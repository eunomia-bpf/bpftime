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
#include "pos/include/handle.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/handle/var.h"
#include "pos/cuda_impl/proto/var.pb.h"


POSHandle_CUDA_Var::POSHandle_CUDA_Var(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Var;
}


POSHandle_CUDA_Var::POSHandle_CUDA_Var(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Var;
}


POSHandle_CUDA_Var::POSHandle_CUDA_Var(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Var::tear_down(){
    return POS_FAILED_NOT_IMPLEMENTED;
}



pos_retval_t POSHandle_CUDA_Var::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Var::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Var *cuda_var_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_var_binary = new pos_protobuf::Bin_POSHandle_CUDA_Var();
    POS_CHECK_POINTER(cuda_var_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_var_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_var_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    cuda_var_binary->set_global_name(this->global_name);

    return retval;
}


pos_retval_t POSHandle_CUDA_Var::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;
    CUdeviceptr dptr = 0;
    size_t d_size = 0;
    POSHandle *module_handle;

    POS_ASSERT(this->parent_handles.size() == 1);
    POS_CHECK_POINTER(module_handle = this->parent_handles[0]);
    POS_ASSERT(module_handle->resource_type_id = kPOS_ResourceTypeId_CUDA_Module);

    POS_ASSERT(this->global_name.size() > 0);

    cuda_dv_retval = cuModuleGetGlobal(
        &dptr, &d_size, (CUmodule)(module_handle->server_addr), this->global_name.c_str()
    );
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_C_DETAIL("failed to restore CUDA var via cuModuleGetGlobal: retval(%d)", cuda_dv_retval);
        retval = POS_FAILED;
        goto exit;
    }
    this->set_server_addr((void*)dptr);
    this->mark_status(kPOS_HandleStatus_Active);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Var::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;

    this->_rid = kPOS_ResourceTypeId_CUDA_Var;

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Var::allocate_mocked_resource(
    POSHandle_CUDA_Var** handle,
    std::map<uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *module_handle;
    POS_CHECK_POINTER(handle);

    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Module) == 1);
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Module].size() == 1);
    POS_CHECK_POINTER(module_handle = related_handles[kPOS_ResourceTypeId_CUDA_Module][0]);

    retval = this->__allocate_mocked_resource(handle, size, use_expected_addr, expected_addr, state_size);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA var in the manager");
        goto exit;
    }

    (*handle)->record_parent_handle(module_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Var::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Var::try_restore_from_pool(POSHandle_CUDA_Var* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Var::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Var** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Var cuda_var_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_var_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_var_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_var_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_var_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_var_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_var_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_var_binary.mutable_base()->id(),
        /* client_addr */ cuda_var_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_var_binary.mutable_base()->server_addr(),
        /* size */ cuda_var_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_var_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_var_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

    (*handle)->global_name = cuda_var_binary.global_name();

exit:
    return retval;
}
