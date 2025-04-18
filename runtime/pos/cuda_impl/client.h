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
#include <filesystem>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/client.h"
#include "pos/include/transport.h"

#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/api_index.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/worker.h"


/*!
 *  \brief  context of CUDA client
 */
typedef struct pos_client_cxt_CUDA {
    POS_CLIENT_CXT_HEAD;
} pos_client_cxt_CUDA_t;


class POSClient_CUDA : public POSClient {
    /* ====================== basic ====================== */
 public:
    /*!
     *  \brief  constructor
     *  \param  id  client identifier
     *  \param  cxt context to initialize this client
     */
    POSClient_CUDA(pos_client_uuid_t id, pid_t pid, pos_client_cxt_CUDA_t cxt, POSWorkspace *ws);
    POSClient_CUDA();
    
    
    /*!
     *  \brief  deconstructor
     */
    ~POSClient_CUDA();
    

    /*!
     *  \brief  instantiate handle manager for all used resources
     *  \note   the children class should replace this method to initialize their 
     *          own needed handle managers
     *  \param  is_restoring    identify whether we're restoring a client, if it's, 
     *                          we won't initialize initial handles inside each 
     *                          handle manager
     *  \return POS_SUCCESS for successfully initialization
     */
    pos_retval_t init_handle_managers(bool is_restoring) override;


    /*!
     *  \brief      deinit handle manager for all used resources
     *  \example    CUDA function manager should export the metadata of functions
     */
    void deinit_handle_managers() override;


 private:
    pos_client_cxt_CUDA _cxt_CUDA;
    /* ====================== basic ====================== */


    /* =============== checkpoint / restore ============== */
 public:
    /*!
     *  \brief  persist handle to specific checkpoint files
     *  \note   this function is currently called by the trace system,
     *          normal checkpoint routine would persist handles with
     *          API provided by POSHandle
     *  \param  with_state  whether to persist with handle state
     *  \return POS_SUCCESS for successfully persist
     */
    pos_retval_t persist_handles(bool with_state) override;

 protected:
    /*!
     *  \brief  restore a single handle with specific type
     *  \note   this function is called by POSClient::restore_handles
     *  \param  ckpt_file   path to the checkpoint file of the handle
     *  \param  rid         resource type index of the handle
     *  \param  hid         index of the handle
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t __reallocate_single_handle(const std::string& ckpt_file, pos_resource_typeid_t rid, pos_u64id_t hid) override;


    /*!
     *  \brief  reassign handle's parent from waitlist
     *  \param  handle  pointer to the handle to be processed
     *  \return POS_SUCCESS for successfully reassigned
     */
    pos_retval_t __reassign_handle_parents(POSHandle* handle) override;
    /* =============== checkpoint / restore ============== */


    /* ==================== transport ==================== */
 public:
    /*
     *  \brief  initialization of transport utilities for migration  
     *  \return POS_SUCCESS for successfully initialization
     */
    pos_retval_t init_transport() override;
    /* ==================== transport ==================== */

    
    /* =============== resource management =============== */
 public:
    /*!
     *  \brief  tear down all handles
     *  \note   
     *  \return POS_SUCCESS for successfully tear down
     */
    pos_retval_t tear_down_all_handles() override;

 protected:
    /*!
     *  \brief  obtain all resource type indices of this client
     *  \return all resource type indices of this client
     */
    std::set<pos_resource_typeid_t> __get_resource_idx() override;

 private:
    /*!
     *  \brief  export the metadata of functions
     *  \note   this function is called in deinit_handle_managers
     */
    void __dump_hm_cuda_functions();
    /* =============== resource management =============== */
};
