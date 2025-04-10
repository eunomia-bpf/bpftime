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
#include <string>
#include <cstdlib>

#include <sys/resource.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/handle.h"
#include "pos/cuda_impl/handle.h"


// forward declaration
class POSHandleManager_CUDA_Var;


/*!
 *  \brief  handle for cuda variable
 */
class POSHandle_CUDA_Var final : public POSHandle_CUDA {
 public:
    /*!
     *  \brief  constructor
     *  \param  client_addr     the mocked client-side address of the handle
     *  \param  size_           size of the handle it self
     *  \param  hm              handle manager which this handle belongs to
     *  \param  id_             index of this handle in the handle manager list
     *  \param  state_size_     size of the resource state behind this handle
     */
    POSHandle_CUDA_Var(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0);

    /*!
     *  \param  hm  handle manager which this handle belongs to
     *  \note   this constructor is invoked during restore process, where the content of 
     *          the handle will be resume by deserializing from checkpoint binary
     */
    POSHandle_CUDA_Var(void* hm);

    /*!
     *  \note   never called, just for passing compilation
     */
    POSHandle_CUDA_Var(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0);

    
    /*!
     *  \brief  obtain the resource name begind this handle
     *  \return resource name begind this handle
     */
    std::string get_resource_name(){ return std::string("CUDA Var"); }


    /*!
     *  \brief  tear down the resource behind this handle, recycle it back to handle manager
     *  \note   this function is invoked when a client is dumped, and posd should tear down all resources
     *          it allocates on GPU
     *  \return POS_SUCCESS for successfully tear down
     */
    pos_retval_t tear_down() override;


    /* ======================== handle specific fields ======================= */
 public:
    // name of the global symbol
    std::string global_name;
    /* ======================== handle specific fields ======================= */


    /* ==================== checkpoint add/commit/persist ==================== */
 protected:
    /*!
     *  \brief  add the state of the resource behind this handle to on-device memory
     *  \param  version_id  version of this checkpoint
     *  \param  stream_id   index of the stream to do this checkpoint
     *  \note   the add process must be sync
     *  \return POS_SUCCESS for successfully checkpointed
     */
    pos_retval_t __add(uint64_t version_id, uint64_t stream_id=0) override;


    /*!
     *  \brief  generate protobuf message for this handle
     *  \param  binary      pointer to the generated binary
     *  \param  base_binary pointer to the base field inside the binary
     *  \return POS_SUCCESS for succesfully generation
     */
    pos_retval_t __generate_protobuf_binary(
        google::protobuf::Message** binary,
        google::protobuf::Message** base_binary
    ) override;
    /* ==================== checkpoint add/commit/persist ==================== */


    /* ======================== restore handle & state ======================= */
 protected:
    friend class POSHandleManager_CUDA_Var;
    friend class POSHandleManager<POSHandle_CUDA_Var>;


    /*!
     *  \brief  restore the current handle when it becomes broken state
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t __restore() override;
    /* ======================== restore handle & state ======================= */
};


/*!
 *  \brief   manager for handles of POSHandle_CUDA_Var
 */
class POSHandleManager_CUDA_Var : public POSHandleManager<POSHandle_CUDA_Var> {
 public:
    /*!
     *  \brief  initialize of the handle manager
     *  \note   pre-allocation of handles, e.g., default stream, device, context handles
     *  \param  related_handles related handles to allocate new handles in this manager
     *  \param  is_restoring    is_restoring    identify whether we're restoring a client, if it's, 
     *                          we won't initialize initial handles inside each 
     *                          handle manager
     *  \return POS_SUCCESS for successfully allocation
     */
    pos_retval_t init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring) override;

    /*!
     *  \brief  allocate new mocked CUDA var within the manager
     *  \param  handle          pointer to the mocked handle of the newly allocated resource
     *  \param  related_handles all related handles for helping allocate the mocked resource
     *                          (note: these related handles might be other types)
     *  \param  size            size of the newly allocated resource
     *  \param  use_expected_addr   indicate whether to use expected client-side address
     *  \param  expected_addr   the expected mock addr to allocate the resource (optional)
     *  \param  state_size      size of resource state behind this handle  
     *  \return POS_FAILED_DRAIN for run out of virtual address space; 
     *          POS_SUCCESS for successfully allocation
     */
    pos_retval_t allocate_mocked_resource(
        POSHandle_CUDA_Var** handle,
        std::map</* type */ uint64_t, std::vector<POSHandle*>> related_handles,
        size_t size=kPOS_HandleDefaultSize,
        bool use_expected_addr = false,
        uint64_t expected_addr = 0,
        uint64_t state_size = 0
    ) override;


    /*!
     *  \brief  allocate and restore handles for provision, for fast restore
     *  \param  amount  amount of handles for pooling
     *  \return POS_SUCCESS for successfully preserving
     */
    pos_retval_t preserve_pooled_handles(uint64_t amount) override;


    /*!
     *  \brief  restore handle from pool
     *  \param  handle  the handle to be restored
     *  \return POS_SUCCESS for successfully restoring
     *          POS_FAILED for failed pooled restoring, should fall back to normal path
     */
    pos_retval_t try_restore_from_pool(POSHandle_CUDA_Var* handle) override;


 private:
    /*!
     *  \brief  restore the extra fields of handle with specific type
     *  \note   this function is called by reallocate_single_handle, and implemented by
     *          specific handle type
     *  \param  mapped          mmap handle of the file
     *  \param  ckpt_file_size  size of the checkpoint size (mmap area)
     *  \param  handle          pointer to the restored handle
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t __reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Var** handle) override;
};
