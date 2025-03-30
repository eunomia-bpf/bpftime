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
#include <vector>
#include <algorithm>
#include <filesystem>
#include <string>
#include <fstream>
#include <map>
#include <type_traits>
#include <stdint.h>
#include <assert.h>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/checkpoint.h"
#include "pos/include/proto/handle.pb.h"
#include "google/protobuf/port_def.inc"


pos_retval_t POSHandle::set_passthrough_addr(void *addr, POSHandle* handle_ptr){ 
    using handle_type = typename std::decay<decltype(*this)>::type;

    pos_retval_t retval = POS_SUCCESS;
    client_addr = addr;
    server_addr = addr;
    POSHandleManager<handle_type> *hm_cast = (POSHandleManager<handle_type>*)_hm;

    POS_CHECK_POINTER(hm_cast);
    POS_ASSERT(handle_ptr == this);

    // record client-side address to the map
    retval = hm_cast->record_handle_address(addr, handle_ptr);

exit:
    return retval;
}


void POSHandle::mark_status(pos_handle_status_t status){
    // using handle_type = typename std::decay<decltype(*this)>::type;
    // POSHandleManager<handle_type> *hm_cast = (POSHandleManager<handle_type>*)this->_hm;
    // POS_CHECK_POINTER(hm_cast);

    POSHandleManager<POSHandle>* hm = (POSHandleManager<POSHandle>*)this->_hm;
    POS_CHECK_POINTER(hm);
    hm->mark_handle_status(this, status);
}


void POSHandle::reset_preserve_counter(){ 
    this->_state_preserve_counter.store(0); 
}


bool POSHandle::is_client_addr_in_range(void *addr, uint64_t *offset){
    bool result;

    result = ((uint64_t)client_addr <= (uint64_t)addr) && ((uint64_t)addr < (uint64_t)(client_addr)+size);

    if(result && offset != nullptr){
        *offset = (uint64_t)addr - (uint64_t)client_addr;
    }

    return result;
}


pos_retval_t POSHandle::checkpoint_add(uint64_t version_id, uint64_t stream_id) { 
    pos_retval_t retval = POS_SUCCESS;
    uint8_t old_counter;

    /*!
     *  \brief  [case]  the adding has been finished, nothing need to do
     */
    if(this->_state_preserve_counter >= 2){
        retval = POS_FAILED_ALREADY_EXIST;
        goto exit;
    }

    old_counter = this->_state_preserve_counter.fetch_add(1, std::memory_order_relaxed);
    if (old_counter == 0) {
        /*!
         *  \brief  [case]  no adding on this handle yet, we conduct sync on-device copy from the origin buffer
         *  \note   this process must be sync, as there could have commit process waiting on this adding to be finished
         */
        retval = this->__add(version_id, stream_id);
        this->_state_preserve_counter.store(3, std::memory_order_relaxed);
    } else if (old_counter == 1) {
        /*!
         *  \brief  [case]  there's non-finished adding on this handle, we need to wait until the adding finished
         */
        retval = POS_WARN_ABANDONED;
        while(this->_state_preserve_counter < 3){}
    }

exit:
    return retval;
}


pos_retval_t POSHandle::checkpoint_commit_async(uint64_t version_id, uint64_t stream_id){ 
    pos_retval_t retval = POS_SUCCESS;
    
    #if POS_CONF_EVAL_CkptEnablePipeline == 1
        //  if the on-device cache is enabled, the cache should be added previously by checkpoint_add,
        //  and this commit process doesn't need to be sync, as no ADD could corrupt this process
        retval = this->__commit(version_id, stream_id, /* from_cache */ true, /* is_sync */ false);
    #else
        uint8_t old_counter;
        old_counter = this->_state_preserve_counter.fetch_add(1, std::memory_order_relaxed);
        if (old_counter == 0) {
            /*!
                *  \brief  [case]  no CoW on this handle yet, we directly commit this buffer
                *  \note   the on-device cache is disabled, the commit should comes from the origin buffer, and this
                *          commit must be sync, as there could have CoW waiting on this commit to be finished
                */
            retval = this->__commit(version_id, stream_id, /* from_cache */ false, /* is_sync */ true);
            this->_state_preserve_counter.store(3, std::memory_order_relaxed);
        } else if (old_counter == 1) {
            /*!
                *  \brief  [case]  there's non-finished CoW on this handle, we need to wait until the CoW finished and
                *                  commit from the new buffer
                *  \note   we commit from the cache under this hood, and the commit process is async as there's no CoW 
                *          on this handle anymore
                */
            while(this->_state_preserve_counter < 3){}
            retval = this->__commit(version_id, stream_id, /* from_cache */ true, /* is_sync */ false);
        } else {
            /*!
                *  \brief  [case]  there's finished CoW on this handle, we can directly commit from the cache
                *  \note   same as the last case
                */
            retval = this->__commit(version_id, stream_id, /* from_cache */ true, /* is_sync */ false);
        }
    #endif  // POS_CONF_EVAL_CkptEnablePipeline        
    
    return retval;
}


pos_retval_t POSHandle::checkpoint_commit_sync(uint64_t version_id, uint64_t stream_id) {
    return this->__commit(version_id, stream_id, /* from_cache */ false, /* is_sync */ true);
}


pos_retval_t POSHandle::checkpoint_commit_host(uint64_t version_id, void* data, uint64_t size){
    pos_retval_t retval = POS_SUCCESS;
    POSCheckpointSlot *ckpt_slot = nullptr;
    
    POS_CHECK_POINTER(data);
    POS_ASSERT(size > 0);

    // apply new host-side checkpoint slot for host-side state
    if(unlikely(POS_SUCCESS != (
        this->ckpt_bag->template apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(
            /* version */ version_id,
            /* ptr */ &ckpt_slot,
            /* dynamic_state_size */ size,
            /* force_overwrite */ false
        )
    ))){
        POS_WARN_C("failed to apply host-side checkpoint slot");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(ckpt_slot);

    memcpy(ckpt_slot->expose_pointer(), data, size);

exit:
    return retval;
}


pos_retval_t POSHandle::sync_persist(){
    pos_retval_t retval = POS_SUCCESS;
    std::future<pos_retval_t> persist_future;

    if(this->_persist_thread != nullptr){
        POS_ASSERT(this->_persist_promise != nullptr);
        persist_future = this->_persist_promise->get_future();
        persist_future.wait();
        retval = persist_future.get();

        // ref: https://en.cppreference.com/w/cpp/thread/thread/%7Ethread
        if(this->_persist_thread->joinable()){
            this->_persist_thread->join();
        }

        delete this->_persist_thread;
        this->_persist_thread = nullptr;
        delete this->_persist_promise;
        this->_persist_promise = nullptr;

        POS_DEBUG("persist thread finished: hid(%lu), retval(%d)", this->id, retval);
    } else {
        retval = POS_FAILED_NOT_EXIST;
    }
    
exit:
    return retval;
}


pos_retval_t POSHandle::checkpoint_persist_async(std::string ckpt_dir, bool with_state, uint64_t version_id){
    pos_retval_t retval = POS_SUCCESS, prev_retval;
    std::future<pos_retval_t> persist_future;
    POSCheckpointSlot *ckpt_slot = nullptr;

    POS_ASSERT(ckpt_dir.size() > 0);

    // verify the path exists
    if(unlikely(!std::filesystem::exists(ckpt_dir))){
        POS_WARN_C(
            "failed to persist checkpoint, no ckpt directory exists, this is a bug: ckpt_dir(%s)",
            ckpt_dir.c_str()
        );
        retval = POS_FAILED_NOT_EXIST;
        goto exit;
    }

    // try obtain checkpoint slot if needed
    // if the handle isn't active, we will persist without state
    if(with_state == true && this->status == kPOS_HandleStatus_Active){
        retval = this->__get_checkpoint_slot_for_persist(&ckpt_slot, version_id);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN_C(
                "failed persist, failed to get checkpoint slot with specified version: hid(%id), ckpt_version(%lu)",
                this->id, version_id
            );
            goto exit;
        }
        POS_CHECK_POINTER(ckpt_slot);
    }

    // collect previous persisting thread if any
    if(this->_persist_thread != nullptr){
        if(unlikely(POS_SUCCESS != (prev_retval = this->sync_persist()))){
            POS_WARN_C("pervious handle persisting is failed: hid(%lu), retval(%u)", this->id, prev_retval);
        }
    }

    this->_persist_promise = new std::promise<pos_retval_t>;
    POS_CHECK_POINTER(this->_persist_promise);

    // persist asynchronously
    this->_persist_thread = new std::thread(
        [](POSHandle* handle, POSCheckpointSlot* ckpt_slot, std::string ckpt_dir){
            pos_retval_t retval = handle->__persist_async_thread(ckpt_slot, ckpt_dir);
            handle->_persist_promise->set_value(retval);
        },
        this, ckpt_slot, ckpt_dir
    );
    POS_CHECK_POINTER(this->_persist_thread);

    POS_DEBUG(
        "persist thread started: hid(%lu), with_state(%s), ckpt_dir(%s)",
        this->id,
        ckpt_slot != nullptr ? "true" : "false",
        ckpt_dir.c_str()
    );

exit:
    return retval;
}


pos_retval_t POSHandle::checkpoint_persist_sync(std::string ckpt_dir, bool with_state, uint64_t version_id){
    pos_retval_t retval = POS_SUCCESS;

    // raise persist thread
    retval = this->checkpoint_persist_async(ckpt_dir, with_state, version_id);
    if(unlikely(retval != POS_SUCCESS)){
        goto exit;
    }

    // sync the persist thread
    retval = this->sync_persist();
    if(unlikely(retval != POS_SUCCESS)){
        goto exit;
    }

exit:
    return retval;
}


pos_retval_t POSHandle::__persist_async_thread(POSCheckpointSlot* ckpt_slot, std::string ckpt_dir){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i, actual_state_size;
    std::string ckpt_file_path;
    std::ofstream ckpt_file_stream;
    google::protobuf::Message *handle_binary = nullptr, *_base_binary = nullptr;
    pos_protobuf::Bin_POSHandle *base_binary = nullptr;

    POS_ASSERT(std::filesystem::exists(ckpt_dir));

    if(unlikely(POS_SUCCESS != (
        retval = this->__generate_protobuf_binary(&handle_binary, &_base_binary)
    ))){
        POS_WARN_C("failed to generate protobuf binary: server_addr(%p), retval(%u)",
            this->server_addr, retval
        );
        goto exit;
    }
    POS_CHECK_POINTER(handle_binary);
    POS_CHECK_POINTER(base_binary = reinterpret_cast<pos_protobuf::Bin_POSHandle*>(_base_binary));

    // ==================== 1. base fields ====================
    base_binary->set_id(this->id);
    base_binary->set_resource_type_id(this->resource_type_id);
    base_binary->set_client_addr((uint64_t)(this->client_addr));
    base_binary->set_server_addr((uint64_t)(this->server_addr));
    base_binary->set_size(this->size);

    // ==================== 2. parent information ====================
    for(i=0; i<parent_handles.size(); i++){
        POS_CHECK_POINTER(this->parent_handles[i]);
        base_binary->add_parent_handle_resource_type_idx(this->parent_handles[i]->resource_type_id);
        base_binary->add_parent_handle_idx(this->parent_handles[i]->id);
    }

    // ==================== 3. state ====================
    if(ckpt_slot != nullptr){
        // TODO: we must ensure the ckpt_slot won't be released until this ckpt ends!
        //      we haven't do that!

        //! \note   we adopt state size inside ckpt slot first, as we might persiting a host-side state
        //          that have dynamic state size that not recorded inside the handle
        actual_state_size = ckpt_slot->get_state_size();
    } else {
        actual_state_size = this->state_size;
    }
    base_binary->set_state_size(actual_state_size);
    if(unlikely(actual_state_size > 0 && ckpt_slot == nullptr)){
        //! \note   this is allowed after we introduce trace system 
        // POS_ERROR_C("serialize stateful handle without providing checkpoint slot, this is a bug");
    }
    
    if(ckpt_slot != nullptr){
        base_binary->set_state_type(static_cast<uint32_t>(ckpt_slot->state_type));
        base_binary->set_state(reinterpret_cast<const char*>(ckpt_slot->expose_pointer()), actual_state_size);
    }

    // form the path to the checkpoint file of this handle
    ckpt_file_path = ckpt_dir 
                    + std::string("/h-")
                    + std::to_string(this->resource_type_id) 
                    + std::string("-")
                    + std::to_string(this->id)
                    + std::string(".bin");

    // write to file
    ckpt_file_stream.open(ckpt_file_path, std::ios::binary | std::ios::out);
    if(!ckpt_file_stream){
        POS_WARN_C(
            "failed to dump checkpoint to file, failed to open file: path(%s)",
            ckpt_file_path.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }
    if(!handle_binary->SerializeToOstream(&ckpt_file_stream)){
        POS_WARN_C(
            "failed to dump checkpoint to file, protobuf failed to dump: path(%s)",
            ckpt_file_path.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }

exit:
    if(ckpt_file_stream.is_open()){ ckpt_file_stream.close(); }
    return retval;
}


pos_retval_t POSHandle::restore() {
    using handle_type = typename std::decay<decltype(*this)>::type;

    pos_retval_t retval;
    POSHandleManager<handle_type> *hm_cast = (POSHandleManager<handle_type>*)this->_hm;

    #if POS_CONF_EVAL_RstEnableContextPool == 1
        retval = hm_cast->try_restore_from_pool(this);
        if(likely(retval == POS_SUCCESS)){
            goto exit;
        }
    #endif // POS_CONF_EVAL_RstEnableContextPool

    retval = this->__restore(); 

exit:
    return retval;
}


pos_retval_t POSHandle::reload_state(uint64_t stream_id){
    pos_retval_t retval = POS_FAILED_NOT_EXIST;

    POS_ASSERT(this->state_size > 0);
    POS_CHECK_POINTER(this->restore_binary_mapped);
    POS_ASSERT(this->restore_binary_mapped_size > 0);

    if(unlikely(this->status != kPOS_HandleStatus_Active)){
        POS_WARN(
            "failed to reload handle state as the handle isn't active yet: server_addr(%p), status(%d)",
            this->server_addr, this->status
        );
        retval = POS_FAILED;
        goto exit;
    }

    return this->__reload_state(
        /* mapped */ this->restore_binary_mapped,
        /* ckpt_file_size */ this->restore_binary_mapped_size,
        /* stream_id */ stream_id
    );

exit:
    return retval;
}


void POSHandle::collect_broken_handles(pos_broken_handle_list_t *broken_handle_list, uint16_t layer_id){
    uint64_t i;

    POS_CHECK_POINTER(broken_handle_list);

    // insert itself to the nonactive_handles map if itsn't active
    if(unlikely(status != kPOS_HandleStatus_Active && status != kPOS_HandleStatus_Delete_Pending)){
        broken_handle_list->add_handle(layer_id, this);
    }

    // iterate over its parent
    for(i=0; i<parent_handles.size(); i++){
        parent_handles[i]->collect_broken_handles(broken_handle_list, layer_id+1);
    }
}
