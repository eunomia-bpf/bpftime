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
#include <set>
#include <map>
#include <unordered_map>

#include <stdint.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/checkpoint.h"
#include "pos/include/utils/timer.h"


POSCheckpointBag::POSCheckpointBag(
    uint64_t fixed_state_size,
    pos_custom_ckpt_allocate_func_t allocator,
    pos_custom_ckpt_deallocate_func_t deallocator,
    pos_custom_ckpt_allocate_func_t dev_allocator,
    pos_custom_ckpt_deallocate_func_t dev_deallocator
) : is_latest_ckpt_finished(false) {
    pos_retval_t tmp_retval;
    uint64_t i=0;
    POSCheckpointSlot *tmp_ptr;

    this->_fixed_state_size = fixed_state_size;
    this->_allocate_func = allocator;
    this->_deallocate_func = deallocator;
    this->_dev_allocate_func = dev_allocator;
    this->_dev_deallocate_func = dev_deallocator;
     
    // preserve host-side checkpoint slot for device state
    if(allocator != nullptr && deallocator != nullptr){
    #define __CKPT_PREFILL_SIZE 1
        for(i=0; i<__CKPT_PREFILL_SIZE; i++){
            tmp_retval = apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
                i, &tmp_ptr, 0, /* force_overwrite */ false
            );
            POS_ASSERT(tmp_retval == POS_SUCCESS);
        }
        for(i=0; i<__CKPT_PREFILL_SIZE; i++){
            tmp_retval = invalidate_by_version<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(i);
            POS_ASSERT(tmp_retval == POS_SUCCESS);
        }
    #undef __CKPT_PREFILL_SIZE
    }

    // preserve device-side checkpoint slot for device state
    if(dev_allocator != nullptr && dev_deallocator != nullptr){
    #define __DEV_CKPT_PREFILL_SIZE 0
        for(i=0; i<__DEV_CKPT_PREFILL_SIZE; i++){
            tmp_retval = apply_checkpoint_slot<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
                i, &tmp_ptr, 0, /* force_overwrite */ false
            );
            POS_ASSERT(tmp_retval == POS_SUCCESS);
        }
        for(i=0; i<__DEV_CKPT_PREFILL_SIZE; i++){
            tmp_retval = invalidate_by_version<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(i);
            POS_ASSERT(tmp_retval == POS_SUCCESS);
        }
    #undef __DEV_CKPT_PREFILL_SIZE
    }
}


/*!
 *  \brief  clear current checkpoint bag
 */
void POSCheckpointBag::clear(){
    typename std::unordered_map<uint64_t, POSCheckpointSlot*>::iterator map_iter;

    for(map_iter = _dev_state_host_slot_map.begin(); map_iter != _dev_state_host_slot_map.end(); map_iter++){
        if(likely(map_iter->second != nullptr)){
            delete map_iter->second;
        }
    }

    for(map_iter = _cached_dev_state_host_slot_map.begin(); map_iter != _cached_dev_state_host_slot_map.end(); map_iter++){
        if(likely(map_iter->second != nullptr)){
            delete map_iter->second;
        }
    }

    _dev_state_host_slot_map.clear();
    _cached_dev_state_host_slot_map.clear();
    _dev_state_host_slot_version_set.clear();
}


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
pos_retval_t POSCheckpointBag::apply_checkpoint_slot(
    uint64_t version, POSCheckpointSlot** ptr, uint64_t dynamic_state_size, bool force_overwrite
){
    pos_retval_t retval = POS_SUCCESS;
    typename std::unordered_map<uint64_t, POSCheckpointSlot*>::iterator map_iter;
    uint64_t old_version;
    std::unordered_map<uint64_t, POSCheckpointSlot*> *cached_map, *active_map;
    std::set<uint64_t> *version_set;
    uint64_t state_size;
    pos_custom_ckpt_allocate_func_t allocate_func;
    pos_custom_ckpt_deallocate_func_t deallocate_func;

    // one can't apply a device-side slot to store host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't apply a device-side slot to store host state"
        );
    }

    POS_CHECK_POINTER(ptr);

    state_size = dynamic_state_size > 0 ? dynamic_state_size : this->_fixed_state_size;
    if(unlikely(state_size == 0)){
        POS_WARN_C("failed to apploy checkpoint slot, both dynamic and fixed state size are 0, this is a bug");
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: apply device-side slot for device-side state
            cached_map = &this->_cached_dev_state_dev_slot_map;
            active_map = &this->_dev_state_dev_slot_map;
            version_set = &this->_dev_state_dev_slot_version_set;
            allocate_func = this->_dev_allocate_func;
            deallocate_func = this->_dev_deallocate_func;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: apply host-side slot for device-side state
            cached_map = &this->_cached_dev_state_host_slot_map;
            active_map = &this->_dev_state_host_slot_map;
            version_set = &this->_dev_state_host_slot_version_set;
            allocate_func = this->_allocate_func;
            deallocate_func = this->_deallocate_func;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: apply host-side slot for host-side state
        cached_map = &this->_cached_host_state_host_slot_map;
        active_map = &this->_host_state_host_slot_map;
        version_set = &this->_host_state_host_slot_version_set;
        allocate_func = nullptr;    // the slot will use malloc
        deallocate_func = nullptr;  // the slot will use free
    }

    if(likely(cached_map->size() > 0)){
        // TODO:
        // here we select the oldest one but for device slot for device state we need to 
        // implement a memory allocation mechanism to choose the one with closest size
        // i.e., deal with dynamic_state_size > 0 && force_overwrite == true
        map_iter = cached_map->begin();
        POS_CHECK_POINTER(*ptr = map_iter->second);
        cached_map->erase(map_iter);
    } else {
        if(force_overwrite == true && active_map->size() > 0){
            map_iter = active_map->begin();
            old_version = map_iter->first;
            POS_CHECK_POINTER(*ptr = map_iter->second);
            active_map->erase(map_iter);
            version_set->erase(old_version);
        } else { 
            POS_CHECK_POINTER(*ptr = new POSCheckpointSlot(state_size, allocate_func, deallocate_func, ckpt_slot_pos, ckpt_state_type));
        }
    }
    active_map->insert(std::pair<uint64_t, POSCheckpointSlot*>(version, *ptr));
    version_set->insert(version);

exit:
    return retval;
}
template pos_retval_t POSCheckpointBag::apply_checkpoint_slot<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
    uint64_t version, POSCheckpointSlot** ptr, uint64_t dynamic_state_size, bool force_overwrite
);
template pos_retval_t POSCheckpointBag::apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
    uint64_t version, POSCheckpointSlot** ptr, uint64_t dynamic_state_size, bool force_overwrite
);
template pos_retval_t POSCheckpointBag::apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(
    uint64_t version, POSCheckpointSlot** ptr, uint64_t dynamic_state_size, bool force_overwrite
);


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
pos_retval_t POSCheckpointBag::get_checkpoint_slot(POSCheckpointSlot** ckpt_slot, uint64_t version){
    pos_retval_t retval = POS_SUCCESS;
    std::unordered_map<uint64_t, POSCheckpointSlot*> *active_map;
    std::set<uint64_t> *version_set;

    // one can't obtain a device-side slot that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't obtain a device-side slot that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: get device-side slot for device-side state
            active_map = &this->_dev_state_dev_slot_map;
            version_set = &this->_dev_state_dev_slot_version_set;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: get host-side slot for device-side state
            active_map = &this->_dev_state_host_slot_map;
            version_set = &this->_dev_state_host_slot_version_set;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: get host-side slot for host-side state
        active_map = &this->_host_state_host_slot_map;
        version_set = &this->_host_state_host_slot_version_set;
    }

    if(unlikely(version_set->size() == 0)){
        retval = POS_FAILED_NOT_READY;
        goto exit;
    }
    if(likely(active_map->count(version) > 0)){
        *ckpt_slot = (*active_map)[version];
    } else {
        *ckpt_slot = nullptr;
        retval = POS_FAILED_NOT_EXIST;
    }

exit:
    return retval;
}
template pos_retval_t POSCheckpointBag::get_checkpoint_slot<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
    POSCheckpointSlot** ckpt_slot, uint64_t version
);
template pos_retval_t POSCheckpointBag::get_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
    POSCheckpointSlot** ckpt_slot, uint64_t version
);
template pos_retval_t POSCheckpointBag::get_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(
    POSCheckpointSlot** ckpt_slot, uint64_t version
);


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
pos_retval_t POSCheckpointBag::get_all_scheckpoint_slots(std::vector<POSCheckpointSlot*>& ckpt_slots){
    pos_retval_t retval = POS_SUCCESS;
    std::unordered_map<uint64_t, POSCheckpointSlot*> *active_map;
    typename std::unordered_map<uint64_t, POSCheckpointSlot*>::iterator map_iter;

    // one can't obtain device-side slots that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't obtain device-side slots that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: get device-side slot for device-side state
            active_map = &this->_dev_state_dev_slot_map;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: get host-side slot for device-side state
            active_map = &this->_dev_state_host_slot_map;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: get host-side slot for host-side state
        active_map = &this->_host_state_host_slot_map;
    }

    ckpt_slots.clear();
    for(map_iter = active_map->begin(); map_iter != active_map->end(); map_iter++){
        ckpt_slots.push_back(map_iter->second);
    }

exit:
    return retval;
}
template pos_retval_t POSCheckpointBag::get_all_scheckpoint_slots<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
    std::vector<POSCheckpointSlot*>& ckpt_slots
);
template pos_retval_t POSCheckpointBag::get_all_scheckpoint_slots<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
    std::vector<POSCheckpointSlot*>& ckpt_slots
);
template pos_retval_t POSCheckpointBag::get_all_scheckpoint_slots<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(
    std::vector<POSCheckpointSlot*>& ckpt_slots
);


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
uint64_t POSCheckpointBag::get_nb_checkpoint_slots(){
    std::unordered_map<uint64_t, POSCheckpointSlot*> *active_map;

    // one can't obtain number of device-side slots that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't obtain number of device-side slots that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: get number of device-side slots for device-side state
            active_map = &this->_dev_state_dev_slot_map;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: get number of host-side slots for device-side state
            active_map = &this->_dev_state_host_slot_map;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: get number of host-side slots for host-side state
        active_map = &this->_host_state_host_slot_map;
    }

    return active_map->size();
}
template uint64_t POSCheckpointBag::get_nb_checkpoint_slots<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>();
template uint64_t POSCheckpointBag::get_nb_checkpoint_slots<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>();
template uint64_t POSCheckpointBag::get_nb_checkpoint_slots<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>();


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
std::set<uint64_t> POSCheckpointBag::get_checkpoint_version_set(){
    std::set<uint64_t> *version_set;

    // one can't obtain the version set of device-side slots that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't obtain the version set of device-side slots that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: get version set of device-side slot for device-side state
            version_set = &this->_dev_state_dev_slot_version_set;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: get version set of  host-side slot for device-side state
            version_set = &this->_dev_state_host_slot_version_set;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: get version set of  host-side slot for host-side state
        version_set = &this->_host_state_host_slot_version_set;
    }

    return (*version_set);
}
template std::set<uint64_t> POSCheckpointBag::get_checkpoint_version_set<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>();
template std::set<uint64_t> POSCheckpointBag::get_checkpoint_version_set<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>();
template std::set<uint64_t> POSCheckpointBag::get_checkpoint_version_set<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>();


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
uint64_t POSCheckpointBag::get_memory_consumption(){
    std::unordered_map<uint64_t, POSCheckpointSlot*> *active_map;
    typename std::unordered_map<uint64_t, POSCheckpointSlot*>::iterator map_iter;
    uint64_t size = 0;

    // one can't obtain the size of device-side slots that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't obtain the size of device-side slots that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: get size of device-side slots for device-side state
            active_map = &this->_dev_state_dev_slot_map;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: get size of host-side slots for device-side state
            active_map = &this->_dev_state_host_slot_map;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: get size of host-side slots for host-side state
        active_map = &this->_host_state_host_slot_map;
    }

    for(map_iter = active_map->begin(); map_iter != active_map->end(); map_iter++){
        POS_CHECK_POINTER(map_iter->second);
        size += map_iter->second->get_state_size();
    }

    return size;
}
template uint64_t POSCheckpointBag::get_memory_consumption<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>();
template uint64_t POSCheckpointBag::get_memory_consumption<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>();
template uint64_t POSCheckpointBag::get_memory_consumption<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>();


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
pos_retval_t POSCheckpointBag::invalidate_by_version(uint64_t version) {
    pos_retval_t retval = POS_SUCCESS;
    POSCheckpointSlot *ckpt_slot;
    std::unordered_map<uint64_t, POSCheckpointSlot*> *cached_map, *active_map;
    std::set<uint64_t> *version_set;

    // one can't invalidate a device-side slot that stores host state
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Host){
        static_assert(
            ckpt_slot_pos != kPOS_CkptSlotPosition_Device,
            "one can't invalidate a device-side slot that stores host state"
        );
    }

    // obtain corresponding map and set
    if constexpr (ckpt_state_type == kPOS_CkptStateType_Device){
        if constexpr (ckpt_slot_pos == kPOS_CkptSlotPosition_Device){
            // case: invalidate device-side slot for device-side state
            cached_map = &this->_cached_dev_state_dev_slot_map;
            active_map = &this->_dev_state_dev_slot_map;
            version_set = &this->_dev_state_dev_slot_version_set;
        } else { // ckpt_slot_pos == kPOS_CkptSlotPosition_Host
            // case: invalidate host-side slot for device-side state
            cached_map = &this->_cached_dev_state_host_slot_map;
            active_map = &this->_dev_state_host_slot_map;
            version_set = &this->_dev_state_host_slot_version_set;
        }
    } else { // ckpt_state_type == kPOS_CkptStateType_Host
        // case: invalidate host-side slot for host-side state
        cached_map = &this->_cached_host_state_host_slot_map;
        active_map = &this->_host_state_host_slot_map;
        version_set = &this->_host_state_host_slot_version_set;
    }

    // check whether checkpoint exit
    retval = this->get_checkpoint_slot<ckpt_slot_pos, ckpt_state_type>(&ckpt_slot, version);
    if(POS_SUCCESS != retval){
        goto exit;
    }
    POS_CHECK_POINTER(ckpt_slot);

    active_map->erase(version);
    version_set->erase(version);
    cached_map->insert(std::pair<uint64_t,POSCheckpointSlot*>(version, ckpt_slot));

exit:
    return retval;
}
template pos_retval_t POSCheckpointBag::invalidate_by_version<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(uint64_t version);
template pos_retval_t POSCheckpointBag::invalidate_by_version<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(uint64_t version);
template pos_retval_t POSCheckpointBag::invalidate_by_version<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>(uint64_t version);


template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
pos_retval_t POSCheckpointBag::invalidate_all_version(){
    pos_retval_t retval = POS_SUCCESS;
    std::set<uint64_t> version_set;
    typename std::set<uint64_t>::iterator version_set_iter;

    version_set = this->get_checkpoint_version_set<ckpt_slot_pos, ckpt_state_type>();
    for(version_set_iter = version_set.begin(); version_set_iter != version_set.end(); version_set_iter++){
        retval = this->invalidate_by_version<ckpt_slot_pos, ckpt_state_type>(*version_set_iter);
        if(unlikely(retval != POS_SUCCESS)){
            goto exit;
        }
    }
    
exit:
    return retval;
}
template pos_retval_t POSCheckpointBag::invalidate_all_version<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>();
template pos_retval_t POSCheckpointBag::invalidate_all_version<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>();
template pos_retval_t POSCheckpointBag::invalidate_all_version<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Host>();


pos_retval_t POSCheckpointBag::load(uint64_t version, void* ckpt_data){
    pos_retval_t retval = POS_SUCCESS;
    POSCheckpointSlot *ckpt_slot;

    POS_CHECK_POINTER(ckpt_data);

    retval = apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
        version, &ckpt_slot, 0, /* force_overwrite */ false
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to apply new checkpoiont slot while loading in restore: version(%lu)", version);
        goto exit;
    }
    POS_CHECK_POINTER(ckpt_slot);

    memcpy(ckpt_slot->expose_pointer(), ckpt_data, this->_fixed_state_size);

exit:
    return retval;
}
