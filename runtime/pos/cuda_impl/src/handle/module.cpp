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
#include <map>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/handle/module.h"
#include "pos/cuda_impl/proto/module.pb.h"


POSHandle_CUDA_Module::POSHandle_CUDA_Module(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Module;

    // initialize checkpoint bag
    #if POS_CONF_EVAL_CkptOptLevel > 0 || POS_CONF_EVAL_MigrOptLevel > 0
        if(unlikely(POS_SUCCESS != this->__init_ckpt_bag())){
            POS_ERROR_C_DETAIL("failed to inilialize checkpoint bag");
        }
    #endif
}


POSHandle_CUDA_Module::POSHandle_CUDA_Module(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Module;
}


POSHandle_CUDA_Module::POSHandle_CUDA_Module(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Module::tear_down(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;

    if(unlikely(this->status != kPOS_HandleStatus_Active)){ goto exit; }

    cuda_dv_retval = cuModuleUnload((CUmodule)(this->server_addr));
    if(unlikely(cuda_dv_retval != CUDA_SUCCESS)){
        POS_WARN_C(
            "failed to tear down CUDA module: id(%lu), client_addr(%p), server_addr(%p)",
            this->id, this->client_addr, this->server_addr
        );
        retval = POS_FAILED;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Module::__init_ckpt_bag(){ 
    this->ckpt_bag = new POSCheckpointBag(
        /* fixed_state_size */ this->state_size,
        /* allocator */ nullptr,
        /* deallocator */ nullptr,
        /* dev_allocator */ nullptr,
        /* dev_deallocator */ nullptr
    );
    POS_CHECK_POINTER(this->ckpt_bag);
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Module::__add(uint64_t version_id, uint64_t stream_id){
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Module::__commit(uint64_t version_id, uint64_t stream_id, bool from_cache, bool is_sync){
    /* nothing to be commited, its state is on host-side */
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Module::__get_checkpoint_slot_for_persist(POSCheckpointSlot** ckpt_slot, uint64_t version_id){
    pos_retval_t retval = POS_SUCCESS;
    std::vector<POSCheckpointSlot*> ckpt_slots;

    POS_CHECK_POINTER(ckpt_slot);

    if(unlikely(POS_SUCCESS != (
        retval = this->ckpt_bag->get_all_scheckpoint_slots<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(ckpt_slots)
    ))){
        POS_WARN_C("failed to obtain host-side checkpoint slot that stores host-side state");
        goto exit;
    }
    POS_ASSERT(ckpt_slots.size() == 1);

    POS_CHECK_POINTER(*ckpt_slot = ckpt_slots[0]);

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Module::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Module *cuda_module_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_module_binary = new pos_protobuf::Bin_POSHandle_CUDA_Module();
    POS_CHECK_POINTER(cuda_module_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_module_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_module_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_CUDA_Module::__restore(){
    pos_retval_t retval = POS_SUCCESS;

    /*!
     *  \note   module is kind of special, it would immediately load its state once created,
     *          so we would actually restore module in __reload_state function
     */
    this->mark_status(kPOS_HandleStatus_Active);
    this->mark_state_status(kPOS_HandleStatus_StateMiss);

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Module::__reload_state(void* mapped, uint64_t ckpt_file_size, uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;
    pos_protobuf::Bin_POSHandle_CUDA_Module module_binary;
    CUmodule module = NULL;

    POS_CHECK_POINTER(mapped);

    if(!module_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle state, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(module_binary.mutable_base());

    #if POS_CONF_RUNTIME_EnableTrace
        ((POSHandleManager_CUDA_Module*)(this->_hm))->metric_tickers.start(POSHandleManager_CUDA_Module::RESTORE_reload_state);
    #endif

    cuda_dv_retval = cuModuleLoadData(
        /* module */ &module,
        /* image */  reinterpret_cast<const void*>(module_binary.mutable_base()->state().c_str())
    );
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_C_DETAIL("failed to restore CUDA module, cuModuleLoadData failed: %d", cuda_dv_retval);
        retval = POS_FAILED;
        goto exit;
    }

    #if POS_CONF_RUNTIME_EnableTrace
        ((POSHandleManager_CUDA_Module*)(this->_hm))->metric_tickers.end(POSHandleManager_CUDA_Module::RESTORE_reload_state);
        ((POSHandleManager_CUDA_Module*)(this->_hm))->metric_reducers.reduce(
            POSHandleManager_CUDA_Module::RESTORE_nb_reload_functions,
            this->function_desps.size()
        );
    #endif

    this->set_server_addr((void*)module);
    this->mark_status(kPOS_HandleStatus_Active);
    this->mark_state_status(kPOS_HandleStatus_StateReady);

exit:
    // this should be the end of using this mmap area, so we release it here
    munmap(mapped, ckpt_file_size);
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Module::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;

    this->_rid = kPOS_ResourceTypeId_CUDA_Module;

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Module::load_cached_function_metas(std::string &file_path){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i;
    std::string line, stream;
    POSCudaFunctionDesp_t *new_desp;
    char delimiter = '|';

    auto generate_desp_from_meta = [](std::vector<std::string>& metas) -> POSCudaFunctionDesp_t* {
        uint64_t i;
        std::vector<uint32_t> param_offsets;
        std::vector<uint32_t> param_sizes;
        std::vector<uint32_t> input_pointer_params;
        std::vector<uint32_t> output_pointer_params;
        std::vector<uint32_t> inout_pointer_params;
        std::vector<uint32_t> suspicious_params;
        std::vector<std::pair<uint32_t,uint64_t>> confirmed_suspicious_params;
        bool confirmed;
        uint64_t nb_input_pointer_params, nb_output_pointer_params, nb_inout_pointer_params, 
                nb_suspicious_params, nb_confirmed_suspicious_params, has_verified_params;
        uint64_t ptr;

        POSCudaFunctionDesp_t *new_desp = new POSCudaFunctionDesp_t();
        POS_CHECK_POINTER(new_desp);

        ptr = 0;
        
        // mangled name of the kernel
        new_desp->name = metas[ptr];
        ptr++;

        // signature of the kernel
        new_desp->signature = metas[ptr];
        ptr++;

        // number of paramters
        new_desp->nb_params = std::stoul(metas[ptr]);
        ptr++;

        // parameter offsets
        for(i=0; i<new_desp->nb_params; i++){
            param_offsets.push_back(std::stoul(metas[ptr+i]));   
        }
        ptr += new_desp->nb_params;
        new_desp->param_offsets = param_offsets;

        // parameter sizes
        for(i=0; i<new_desp->nb_params; i++){
            param_sizes.push_back(std::stoul(metas[ptr+i]));
        }
        ptr += new_desp->nb_params;
        new_desp->param_sizes = param_sizes;

        // input paramters
        nb_input_pointer_params = std::stoul(metas[ptr]);
        ptr++;
        for(i=0; i<nb_input_pointer_params; i++){
            input_pointer_params.push_back(std::stoul(metas[ptr+i]));
        }
        ptr += nb_input_pointer_params;
        new_desp->input_pointer_params = input_pointer_params;

        // output paramters
        nb_output_pointer_params = std::stoul(metas[ptr]);
        ptr++;
        for(i=0; i<nb_output_pointer_params; i++){
            output_pointer_params.push_back(std::stoul(metas[ptr+i]));
        }
        ptr += nb_output_pointer_params;
        new_desp->output_pointer_params = output_pointer_params;

        // inout paramters
        nb_inout_pointer_params = std::stoul(metas[ptr]);
        ptr++;
        for(i=0; i<nb_inout_pointer_params; i++){
            inout_pointer_params.push_back(std::stoul(metas[ptr+i]));
        }
        ptr += nb_inout_pointer_params;
        new_desp->inout_pointer_params = inout_pointer_params;

        // suspicious paramters
        nb_suspicious_params = std::stoul(metas[ptr]);
        ptr++;
        for(i=0; i<nb_suspicious_params; i++){
            suspicious_params.push_back(std::stoul(metas[ptr+i]));
        }
        ptr += nb_suspicious_params;
        new_desp->suspicious_params = suspicious_params;

        // has verified suspicious paramters
        has_verified_params = std::stoul(metas[ptr]);
        ptr++;
        new_desp->has_verified_params = has_verified_params;

        if(has_verified_params == 1){
            // index of those parameter which is a structure (contains pointers)
            nb_confirmed_suspicious_params = std::stoul(metas[ptr]);
            ptr++;
            for(i=0; i<nb_confirmed_suspicious_params; i++){
                confirmed_suspicious_params.push_back({
                    /* param_index */ std::stoul(metas[ptr+2*i]), /* offset */ std::stoul(metas[ptr+2*i+1])
                });
            }
            ptr += nb_confirmed_suspicious_params;
            new_desp->confirmed_suspicious_params = confirmed_suspicious_params;
        }

        // cbank parameter size (p.s., what is this?)
        new_desp->cbank_param_size = std::stoul(metas[ptr].c_str());

        return new_desp;
    };

    std::ifstream file(file_path.c_str(), std::ios::in);
    if(likely(file.is_open())){
        POS_LOG("parsing cached kernel metas from file %s...", file_path.c_str());
        i = 0;
        while (std::getline(file, line)) {
            // split by ","
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> metas;
            while (std::getline(ss, segment, delimiter)) { metas.push_back(segment); }

            // parse
            new_desp = generate_desp_from_meta(metas);
            cached_function_desps.insert(
                std::pair<std::string, POSCudaFunctionDesp_t*>(new_desp->name, new_desp)
            );

            i++;
        }

        POS_LOG("parsed %lu of cached kernel metas from file %s", i, file_path.c_str());
        file.close();
    } else {
        retval = POS_FAILED_NOT_EXIST;
        POS_WARN("failed to load kernel meta file %s, fall back to slow path", file_path.c_str());
    }

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Module::allocate_mocked_resource(
    POSHandle_CUDA_Module** handle,
    std::map</* type */ uint64_t, std::vector<POSHandle*>> related_handles,
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
        POS_WARN_C("failed to allocate mocked CUDA module in the manager");
        goto exit;
    }

    POS_CHECK_POINTER(*handle);
    (*handle)->record_parent_handle(context_handle);

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Module::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Module::try_restore_from_pool(POSHandle_CUDA_Module* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Module::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Module** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Module cuda_module_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_module_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_module_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_module_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_module_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_module_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_module_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_module_binary.mutable_base()->id(),
        /* client_addr */ cuda_module_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_module_binary.mutable_base()->server_addr(),
        /* size */ cuda_module_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_module_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_module_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}


#if POS_CONF_RUNTIME_EnableTrace

void POSHandleManager_CUDA_Module::print_metrics() {
    static std::unordered_map<metrics_ticker_type_t, std::string> ticker_names = {
        { RESTORE_reload_state, "Restore State" }
    };
    static std::unordered_map<metrics_reducer_type_t, std::string> reducer_names = {
        { RESTORE_nb_reload_functions, "# Restored Functions" }
    };
    POS_ASSERT(pos_resource_map.count(this->_rid) > 0);
    POS_LOG(
        "[HandleManager Metrics] %s:\n%s\n%s",
        pos_resource_map[this->_rid].c_str(),
        this->metric_tickers.str(ticker_names).c_str(),
        this->metric_reducers.str(reducer_names).c_str()
    );
}

#endif
