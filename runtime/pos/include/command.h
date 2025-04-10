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
#include <set>
#include "pos/include/common.h"
#include "pos/include/log.h"


// forward declaration
class POSHandle;


/*!
 *  \brief command type index
 */
enum pos_command_typeid_t : uint16_t {
    kPOS_Command_Nothing = 0,

    /* ========== Checkpoint Cmd ========== */
    kPOS_Command_Oob2Parser_PreDump,
    kPOS_Command_Oob2Parser_Dump,
    kPOS_Command_Parser2Worker_PreDump,
    kPOS_Command_Parser2Worker_Dump,
};


/*!
 *  \brief asynchronous command among different threads   
 */
typedef struct POSCommand_QE {
    // type of the command
    pos_command_typeid_t type;

    // client id
    pos_client_uuid_t client_id;

    // command execution result
    pos_retval_t retval;


    // ============================== ckpt payloads ==============================
    // path to store all checkpoints
    std::string ckpt_dir;
    
    // for kPOS_Command_xxx_PreDump and kPOS_Command_xxx_Dump
    std::set<POSHandle*> stateful_handles;
    std::set<POSHandle*> stateless_handles;
    std::set<pos_resource_typeid_t> target_resource_type_idx;
    bool do_cow;
    bool force_recompute;

    /*!
     *  \brief  record all handles that need to be checkpointed within this checkpoint op
     *  \param  handle_set  sets of handles to be added
     *  \param  handle      handle to be added
     */
    inline void record_stateful_handles(std::set<POSHandle*>& handle_set){
        stateful_handles.insert(handle_set.begin(), handle_set.end());
    }
    inline void record_stateful_handles(POSHandle *handle){
        stateful_handles.insert(handle);
    }
    inline void record_stateless_handles(std::set<POSHandle*>& handle_set){
        stateless_handles.insert(handle_set.begin(), handle_set.end());
    }
    inline void record_stateless_handles(POSHandle *handle){
        stateless_handles.insert(handle);
    }
    // ============================== ckpt payloads ==============================

    POSCommand_QE() : type(kPOS_Command_Nothing), retval(POS_SUCCESS) {}
} POSCommand_QE_t;
