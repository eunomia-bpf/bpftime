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
#include "pos/include/checkpoint.h"
#include "pos/include/api_context.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/utils/fatbin.h"


// forward declaration
class POSHandleManager_CUDA_Module;


/*!
 *  \brief  handle for cuda module
 */
class POSHandle_CUDA_Module final : public POSHandle_CUDA {
 public:
    /*!
     *  \brief  constructor
     *  \param  client_addr     the mocked client-side address of the handle
     *  \param  size_           size of the handle it self
     *  \param  hm              handle manager which this handle belongs to
     *  \param  id_             index of this handle in the handle manager list
     *  \param  state_size_     size of the resource state behind this handle
     */
    POSHandle_CUDA_Module(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0);


    /*!
     *  \param  hm  handle manager which this handle belongs to
     *  \note   this constructor is invoked during restore process, where the content of 
     *          the handle will be resume by deserializing from checkpoint binary
     */
    POSHandle_CUDA_Module(void* hm);


    /*!
     *  \note   never called, just for passing compilation
     */
    POSHandle_CUDA_Module(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0);


    /*!
     *  \brief  obtain the resource name begind this handle
     *  \return resource name begind this handle
     */
    std::string get_resource_name(){ return std::string("CUDA Module"); }


    /*!
     *  \brief  tear down the resource behind this handle, recycle it back to handle manager
     *  \note   this function is invoked when a client is dumped, and posd should tear down all resources
     *          it allocates on GPU
     *  \return POS_SUCCESS for successfully tear down
     */
    pos_retval_t tear_down() override;


    /* ======================== handle specific fields ======================= */
 public:
    // function descriptors under this module
    std::vector<POSCudaFunctionDesp*> function_desps;

    // pacthed binary, only PTX included
    // std::vector<uint8_t> patched_binary;

    // shadow module for the patched kernel binary
    // void *patched_server_addr;
    /* ======================== handle specific fields ======================= */


    /* ==================== checkpoint add/commit/persist ==================== */
 protected:
    /*!
     *  \brief  initialize checkpoint bag of this handle
     *  \note   it must be implemented by different implementations of stateful 
     *          handle, as they might require different allocators and deallocators
     *  \return POS_SUCCESS for successfully initialization
     */
    pos_retval_t __init_ckpt_bag() override;


    /*!
     *  \brief  add the state of the resource behind this handle to on-device memory
     *  \param  version_id  version of this checkpoint
     *  \param  stream_id   index of the stream to do this checkpoint
     *  \note   the add process must be sync
     *  \return POS_SUCCESS for successfully checkpointed
     */
    pos_retval_t __add(uint64_t version_id, uint64_t stream_id=0) override;


    /*!
     *  \brief  commit the state of the resource behind this handle
     *  \param  version_id  version of this checkpoint
     *  \param  stream_id   index of the stream to do this checkpoint
     *  \param  from_cow    whether to dump from on-device cow buffer
     *  \param  is_sync    whether the commit process should be sync
     *  \return POS_SUCCESS for successfully checkpointed
     */
    pos_retval_t __commit(
        uint64_t version_id, uint64_t stream_id=0, bool from_cache=false, bool is_sync=false
    ) override;


    /*!
     *  \brief  obtain the checkpoint slot with corresponding version index for persist
     *  \param  ckpt_slot   obtained checkpoint slot
     *  \param  version_id  given version index
     *  \return POS_SUCCESS for successful get
     */
    pos_retval_t __get_checkpoint_slot_for_persist(POSCheckpointSlot** ckpt_slot, uint64_t version_id) override;


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
    friend class POSHandleManager_CUDA_Module;
    friend class POSHandleManager<POSHandle_CUDA_Module>;


    /*!
     *  \brief  restore the current handle when it becomes broken state
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t __restore() override;


    /*!
     *  \brief  reload state of this handle back to the device
     *  \param  mapped          mmap area of the checkpoint file of this handle
     *  \param  ckpt_file_size  size of the checkpoint size (mmap area)
     *  \param  stream_id       stream for reloading the state
     */
    pos_retval_t __reload_state(void* mapped, uint64_t ckpt_file_size, uint64_t stream_id) override;
    /* ======================== restore handle & state ======================= */
};


/*!
 *  \brief   manager for handles of POSHandle_CUDA_Module
 */
class POSHandleManager_CUDA_Module : public POSHandleManager<POSHandle_CUDA_Module> {
 public:
    std::map<std::string, POSCudaFunctionDesp_t*> cached_function_desps;

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
     *  \brief  load kernel metadata which dumps by previous run
     *  \todo   change to use protobuf to store
     *  \param  file_path   path to the file that stores the metadata of kernels
     *  \return POS_SUCCESS for successfully loaded
     */
    pos_retval_t load_cached_function_metas(std::string &file_path);


    /*!
     *  \brief  allocate new mocked CUDA module within the manager
     *  \param  handle              pointer to the mocked handle of the newly allocated resource
     *  \param  related_handles     all related handles for helping allocate the mocked resource
     *                              (note: these related handles might be other types)
     *  \param  size                size of the newly allocated resource
     *  \param  use_expected_addr   indicate whether to use expected client-side address
     *  \param  expected_addr       the expected mock addr to allocate the resource (optional)
     *  \param  state_size          size of resource state behind this handle  
     *  \return POS_FAILED_DRAIN for run out of virtual address space; 
     *          POS_SUCCESS for successfully allocation
     */
    pos_retval_t allocate_mocked_resource(
        POSHandle_CUDA_Module** handle,
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
    pos_retval_t try_restore_from_pool(POSHandle_CUDA_Module* handle) override;


    /* =========================== metric system ============================= */
 public:
    #if POS_CONF_RUNTIME_EnableTrace
        enum metrics_ticker_type_t : uint8_t {
            __TICKER_BASE__ = 0,
            RESTORE_reload_state
        };
        POSMetrics_TickerList<metrics_ticker_type_t> metric_tickers;

        enum metrics_reducer_type_t : uint8_t {
            __REDUCER_BASE__= 0,
            RESTORE_nb_reload_functions
        };
        POSMetrics_ReducerList<metrics_reducer_type_t, uint64_t> metric_reducers;

        void print_metrics() override;
    #endif
    /* =========================== metric system ============================= */


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
    pos_retval_t __reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Module** handle) override;
};
