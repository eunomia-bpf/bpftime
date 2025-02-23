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
#include <vector>
#include <set>
#include <unordered_map>
#include <stdint.h>
#include "pos/include/common.h"
#include "pos/include/log.h"


// forward declaration
typedef struct POSAPIContext_QE POSAPIContext_QE_t;


/*!
 *  \brief  position of the checkpoint slot,
 *          i.e., host/device
 */
enum pos_ckptslot_position_t : uint8_t {
    kPOS_CkptSlotPosition_Device = 0,
    kPOS_CkptSlotPosition_Host
};


/*!
 *  \brief  type of the checkpoint data,
 *          i.e., host/device-side state
 */
enum pos_ckpt_state_type_t : uint8_t {
    kPOS_CkptStateType_Device = 0,
    kPOS_CkptStateType_Host
};


using pos_custom_ckpt_allocate_func_t = void*(*)(uint64_t size);
using pos_custom_ckpt_deallocate_func_t = void(*)(void* ptr);


/*!
 *  \brief  a slot to store one version of checkpoint
 *  \note   don't consider supporting offset in this class!
 */
class POSCheckpointSlot {
 public:
    // position of this checkpoint slot (device/host)
    pos_ckptslot_position_t ckpt_position;


    // type of state stored inside this checkpoint slot (device/host)
    pos_ckpt_state_type_t state_type;


    /*!
     *  \brief  construtor
     *  \param  state_size      size of the data inside this slot
     *  \param  allocator       allocator for allocating host-side memory region to store checkpoint
     *  \param  deallocator     deallocator for deallocating host-side memory region that stores checkpoint
     *  \param  ckpt_position   position of this checkpoint slot (device/host)
     *  \param  state_type      type of state stored inside this checkpoint slot (device/host)
     */
    POSCheckpointSlot(
        uint64_t state_size,
        pos_custom_ckpt_allocate_func_t allocator,
        pos_custom_ckpt_deallocate_func_t deallocator,
        pos_ckptslot_position_t ckpt_position,
        pos_ckpt_state_type_t state_type
    ) : _state_size(state_size),
        _custom_deallocator(deallocator),
        ckpt_position(ckpt_position),
        state_type(state_type)
    {
        POS_ASSERT(state_size > 0);
        if(likely(allocator != nullptr)){
            POS_CHECK_POINTER(this->_data = allocator(state_size));
        } else {
            POS_CHECK_POINTER(this->_data = reinterpret_cast<void*>(new uint8_t[state_size]));
        }
    }

    /*!
     *  \brief  deconstrutor
     */
    ~POSCheckpointSlot(){
        if(likely(_custom_deallocator != nullptr)){
            _custom_deallocator(_data);
        } else {
            // default deallocate using free
            if(unlikely(this->_data != nullptr && this->_state_size > 0)){
                uint8_t *data_u8;
                POS_CHECK_POINTER(data_u8 = reinterpret_cast<uint8_t*>(this->_data));
                delete[] data_u8;
            }
        }
    }

    /*!
     *  \brief  expose the memory pointer of the slot
     *  \return pointer to the checkpoint memory region
     */
    inline void* expose_pointer(){ return this->_data; }

    /*!
     *  \brief  obtain the size of the state in this slot
     *  \return the size of the state in this slot
     */
    inline uint64_t get_state_size(){ return this->_state_size; }

 protected:
    // size of the data inside this slot
    uint64_t _state_size;

    // pointer to the checkpoint memory region
    void *_data;

    // deallocator for deallocating memory region that stores checkpoint
    pos_custom_ckpt_deallocate_func_t _custom_deallocator;
};


/*!
 *  \brief  host-side value checkpoint record
 */
typedef struct pos_host_ckpt {
    // corresponding api context wqe that stores the host-side value
    POSAPIContext_QE_t *wqe;

    // index of the parameter that stores the host-side value
    uint32_t param_index;

    // offset from the base address of the handle
    uint64_t offset;

    // size of this host checkpoint
    uint64_t size;
} pos_host_ckpt_t;


/*!
 *  \brief  collection of checkpoint slots of a handle
 */
class POSCheckpointBag {
 public:
    /*!
     *  \brief  construtor
     *  \param  fixed_state_size    fixed size of the data stored inside this bag
     *  \param  allocator           allocator for allocating host-side memory region to store checkpoint
     *  \param  deallocator         deallocator for deallocating host-side memory region that stores checkpoint
     *  \param  dev_allocator       allocator for allocating device-side memory region to store checkpoint
     *  \param  dev_deallocator     deallocator for deallocating device-side memory region that stores checkpoint
     */
    POSCheckpointBag(
        uint64_t fixed_state_size,
        pos_custom_ckpt_allocate_func_t allocator,
        pos_custom_ckpt_deallocate_func_t deallocator,
        pos_custom_ckpt_allocate_func_t dev_allocator,
        pos_custom_ckpt_deallocate_func_t dev_deallocator
    );
    ~POSCheckpointBag() = default;


    /*!
     *  \brief  clear current checkpoint bag
     */
    void clear();


    /*!
     *  \brief  allocate a new checkpoint slot inside this bag
     *  \tparam ckpt_slot_pos       position of the applied checkpoint slot
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \param  version             version of this checkpoint
     *  \param  ptr                 returned pointer to the checkpoint slot
     *  \param  dynamic_state_size  dynaimc state size, for those resources (e.g., Module) whose
     *                              size can only be determined after the creation of handle; for
     *                              those can be determined at creation, simply set this param as 0
     *  \param  force_overwrite     force to overwrite the oldest checkpoint to save allocation time
     *                              (if no available slot exit)
     *  \return POS_SUCCESS for successfully allocation
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    pos_retval_t apply_checkpoint_slot(
        uint64_t version, POSCheckpointSlot** ptr, uint64_t dynamic_state_size, bool force_overwrite
    );


    /*!
     *  \brief  obtain checkpointed data by given checkpoint version
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be obtained
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \param  ckpt_slot   pointer to the checkpoint slot if successfully obtained
     *  \param  version     the specified version
     *  \return POS_SUCCESS for successfully obtained
     *          POS_FAILED_NOT_EXIST for no checkpoint is found
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    pos_retval_t get_checkpoint_slot(POSCheckpointSlot** ckpt_slot, uint64_t version);


    /*!
     *  \brief  obtain all checkpointed slots
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be obtained
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \param  ckpt_slots          pointer to the vector that stores checkpoint slots
     *  \return POS_SUCCESS for successfully obtained
     *          POS_FAILED_NOT_EXIST for no checkpoint is found
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    pos_retval_t get_all_scheckpoint_slots(std::vector<POSCheckpointSlot*>& ckpt_slots);


    /*!
     *  \brief  obtain the number of recorded checkpoints inside this bag
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be obtained
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \return number of recorded checkpoints
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    uint64_t get_nb_checkpoint_slots();


    /*!
     *  \brief  obtain the checkpoint version list
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be obtained
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \return the checkpoint version list
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    std::set<uint64_t> get_checkpoint_version_set();


    /*!
     *  \brief  invalidate the checkpoint from this bag
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be invalidated
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \param  version version of the checkpoint to be removed
     *  \return POS_SUCCESS for successfully invalidate
     *          POS_NOT_READY for no checkpoint had been record
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    pos_retval_t invalidate_by_version(uint64_t version);


    /*!
     *  \brief  invalidate all checkpoint from this bag
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be invalidated
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \return POS_SUCCESS for successfully invalidate
     *          POS_NOT_READY for no checkpoint had been record
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    pos_retval_t invalidate_all_version();


    /*!
     *  \brief  obtain overall memory consumption of this checkpoint bag
     *  \tparam ckpt_slot_pos       position of the checkpoint slot to be quried
     *  \tparam ckpt_state_type     type of the checkpointed state
     *  \return overall memory consumption of this checkpoint bag
     */
    template<pos_ckptslot_position_t ckpt_slot_pos, pos_ckpt_state_type_t ckpt_state_type>
    uint64_t get_memory_consumption();


    /*!
     *  \brief  load binary checkpoint data into this bag
     *  \note   this function will be invoked during the restore process
     *  \param  version     version of this checkpoint
     *  \param  ckpt_data   pointer to the buffer that stores the checkpointed data
     *  \return POS_SUCCESS for successfully loading
     */
    pos_retval_t load(uint64_t version, void* ckpt_data);


    // indicate whether the checkpoint has been finished in the latest checkpoint round
    bool is_latest_ckpt_finished;

 private:
    /*!
     *  \brief  map of version to host-side checkpoint slot for device state 
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _dev_state_host_slot_map;

    /*!
     *  \brief  map of version to cached host-side checkpoint slot for device state 
     *  \note   we store thost cached version so that we can reuse their memory
     *          space in the next time we apply for a new checkpoint slot
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _cached_dev_state_host_slot_map;

    /*!
     *  \brief  map of version to device-side checkpoint slot for device state 
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _dev_state_dev_slot_map;

    /*!
     *  \brief  map of version to cached device-side checkpoint slot for device state
     *  \note   we store thost cached version so that we can reuse their memory
     *          space in the next time we apply for a new checkpoint slot
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _cached_dev_state_dev_slot_map;

    /*!
     *  \brief  map of version to host-side checkpoint slot for host state 
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _host_state_host_slot_map;

    /*!
     *  \brief  map of version to cached host-side checkpoint slot for host state
     *  \note   we store thost cached version so that we can reuse their memory
     *          space in the next time we apply for a new checkpoint slot
     */
    std::unordered_map<uint64_t, POSCheckpointSlot*> _cached_host_state_host_slot_map;

    // all versions of host-side checkpoint slots that store device state
    std::set<uint64_t> _dev_state_host_slot_version_set;

    // all versions of device-side checkpoint slots that store device state
    std::set<uint64_t> _dev_state_dev_slot_version_set;

    // all versions of host-side checkpoint slots that store host state
    std::set<uint64_t> _host_state_host_slot_version_set;

    // static state size of each checkpoint
    uint64_t _fixed_state_size;

    // allocator and deallocator of checkpoint memory
    pos_custom_ckpt_allocate_func_t _allocate_func;
    pos_custom_ckpt_deallocate_func_t _deallocate_func;

    // allocator and deallocator of on-device checkpoint memory
    pos_custom_ckpt_allocate_func_t _dev_allocate_func;
    pos_custom_ckpt_deallocate_func_t _dev_deallocate_func;

    /*!
     *  \brief  list of host-side checkpoint
     */
    std::unordered_map<uint64_t, pos_host_ckpt_t> _host_ckpt_map;
};
