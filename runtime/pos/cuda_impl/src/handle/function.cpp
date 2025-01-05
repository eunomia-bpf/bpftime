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
#include "pos/cuda_impl/handle/function.h"
#include "pos/cuda_impl/proto/function.pb.h"


POSHandle_CUDA_Function::POSHandle_CUDA_Function(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_) 
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Function;
    this->has_verified_params = false;
}


POSHandle_CUDA_Function::POSHandle_CUDA_Function(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Function;
}


POSHandle_CUDA_Function::POSHandle_CUDA_Function(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Function::tear_down(){
    return POS_FAILED_NOT_IMPLEMENTED;
}


pos_retval_t POSHandle_CUDA_Function::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Function::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i;
    pos_protobuf::Bin_POSHandle_CUDA_Function *cuda_function_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_function_binary = new pos_protobuf::Bin_POSHandle_CUDA_Function();
    POS_CHECK_POINTER(cuda_function_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_function_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_function_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    cuda_function_binary->set_name(this->name);
    cuda_function_binary->set_signature(this->signature);
    cuda_function_binary->set_nb_params(this->nb_params);
    for(i=0; i<this->param_offsets.size(); i++){ cuda_function_binary->add_param_offsets(this->param_offsets[i]); }
    for(i=0; i<this->param_sizes.size(); i++){ cuda_function_binary->add_param_sizes(this->param_sizes[i]); }
    for(i=0; i<this->input_pointer_params.size(); i++){ cuda_function_binary->add_input_pointer_params(this->input_pointer_params[i]); }
    for(i=0; i<this->inout_pointer_params.size(); i++){ cuda_function_binary->add_inout_pointer_params(this->inout_pointer_params[i]); }
    for(i=0; i<this->output_pointer_params.size(); i++){ cuda_function_binary->add_output_pointer_params(this->output_pointer_params[i]); }
    for(i=0; i<this->suspicious_params.size(); i++){ cuda_function_binary->add_suspicious_params(this->suspicious_params[i]); }
    cuda_function_binary->set_has_verified_params(this->has_verified_params);
    for(i=0; i<this->confirmed_suspicious_params.size(); i++){
        pos_protobuf::Bin_Suspicious_Param_Pair* pair = cuda_function_binary->add_confirmed_suspicious_params();
        pair->set_index(this->confirmed_suspicious_params[i].first);
        pair->set_offset(this->confirmed_suspicious_params[i].second);
    }
    cuda_function_binary->set_cbank_param_size(this->cbank_param_size);

    return retval;
}


pos_retval_t POSHandle_CUDA_Function::__restore(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;
    CUfunction function = NULL;
    POSHandle *module_handle;

    POS_ASSERT(this->parent_handles.size() == 1);
    POS_CHECK_POINTER(module_handle = this->parent_handles[0]);
    POS_ASSERT(module_handle->resource_type_id = kPOS_ResourceTypeId_CUDA_Module);
    
    POS_ASSERT(this->name.size() > 0);

    cuda_dv_retval = cuModuleGetFunction(
        &function, (CUmodule)(module_handle->server_addr), this->name.c_str()
    );

    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_C_DETAIL("failed to restore CUDA function: retval(%d)", cuda_dv_retval);
        retval = POS_FAILED;
        goto exit;
    }

    this->set_server_addr((void*)function);
    this->mark_status(kPOS_HandleStatus_Active);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Function::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;

    this->_rid = kPOS_ResourceTypeId_CUDA_Function;

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Function::allocate_mocked_resource(
    POSHandle_CUDA_Function** handle,
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

    module_handle = related_handles[kPOS_ResourceTypeId_CUDA_Module][0];
    POS_CHECK_POINTER(module_handle);

    // old: when allocate CUDA function, we would use expected client-side address
    // new: not true, consider cuModuleGetFunction
    // POS_ASSERT(use_expected_addr == true);

    retval = this->__allocate_mocked_resource(handle, size, use_expected_addr, expected_addr, state_size);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA var in the manager");
        goto exit;
    }
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA function in the manager");
        goto exit;
    }

    (*handle)->record_parent_handle(module_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Function::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Function::try_restore_from_pool(POSHandle_CUDA_Function* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Function::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Function** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Function cuda_function_binary;
    pos_protobuf::Bin_Suspicious_Param_Pair sus_param_pair;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_function_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_function_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_function_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_function_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_function_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_function_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_function_binary.mutable_base()->id(),
        /* client_addr */ cuda_function_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_function_binary.mutable_base()->server_addr(),
        /* size */ cuda_function_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_function_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_function_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

    (*handle)->name = cuda_function_binary.name();
    (*handle)->signature = cuda_function_binary.signature();
    (*handle)->nb_params = cuda_function_binary.nb_params();

    for (i=0; i<cuda_function_binary.param_offsets_size(); i++) {
        (*handle)->param_offsets.push_back(cuda_function_binary.param_offsets(i));
    }
    for (i=0; i<cuda_function_binary.param_sizes_size(); i++) {
        (*handle)->param_sizes.push_back(cuda_function_binary.param_sizes(i));
    }
    for (i=0; i<cuda_function_binary.input_pointer_params_size(); i++) {
        (*handle)->input_pointer_params.push_back(cuda_function_binary.input_pointer_params(i));
    }
    for (i=0; i<cuda_function_binary.inout_pointer_params_size(); i++) {
        (*handle)->inout_pointer_params.push_back(cuda_function_binary.inout_pointer_params(i));
    }
    for (i=0; i<cuda_function_binary.output_pointer_params_size(); i++) {
        (*handle)->output_pointer_params.push_back(cuda_function_binary.output_pointer_params(i));
    }
    for (i=0; i<cuda_function_binary.suspicious_params_size(); i++) {
        (*handle)->suspicious_params.push_back(cuda_function_binary.suspicious_params(i));
    }

    (*handle)->has_verified_params = cuda_function_binary.has_verified_params();

    for (i=0; i<cuda_function_binary.confirmed_suspicious_params_size(); i++) {
        sus_param_pair = cuda_function_binary.confirmed_suspicious_params(i);
        (*handle)->confirmed_suspicious_params.push_back(
            std::pair<pos_resource_typeid_t, pos_u64id_t>(sus_param_pair.index(), sus_param_pair.offset())
        );
    }

    (*handle)->cbank_param_size = cuda_function_binary.cbank_param_size();

exit:
    return retval;
}
