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


/*!
 *  \brief  idx of CUDA resource types
 */
enum : pos_resource_typeid_t {
    kPOS_ResourceTypeId_CUDA_Context = kPOS_ResourceTypeId_Num_Base_Type,
    kPOS_ResourceTypeId_CUDA_Module,
    kPOS_ResourceTypeId_CUDA_Function,
    kPOS_ResourceTypeId_CUDA_Var,
    kPOS_ResourceTypeId_CUDA_Device,
    kPOS_ResourceTypeId_CUDA_Memory,
    kPOS_ResourceTypeId_CUDA_Stream,
    kPOS_ResourceTypeId_CUDA_Event,

    /*! \note   library handle types, define in pos/cuda_impl/handle/xxx.h */
    kPOS_ResourceTypeId_cuBLAS_Context
};


/*!
 *  \brief  handle for cuda structs
 */
class POSHandle_CUDA : public POSHandle {
 public:
    /*!
     *  \brief  constructor
     *  \param  size_           size of the handle it self
     *  \param  hm              handle manager which this handle belongs to
     *  \param  id_             index of this handle in the handle manager list
     *  \param  state_size_     size of the resource state behind this handle
     */
    POSHandle_CUDA(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0)
        : POSHandle(size_, hm, id_, state_size_){}

    /*!
     *  \brief  constructor
     *  \param  hm  handle manager which this handle belongs to
     *  \note   this constructor is invoked during restore process, where the content of 
     *          the handle will be resume by deserializing from checkpoint binary
     */
    POSHandle_CUDA(void* hm) : POSHandle(hm){}

    /*!
     *  \brief  constructor
     *  \param  client_addr     the mocked client-side address of the handle
     *  \param  size_           size of the handle it self
     *  \param  hm              handle manager which this handle belongs to
     *  \param  id_             index of this handle in the handle manager list
     *  \param  state_size_     size of the resource state behind this handle
     */
    POSHandle_CUDA(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_=0)
        : POSHandle(client_addr_, size_, hm, id_, state_size_){}

    /* ===================== platform-specific functions ===================== */
 protected:
    /*!
     *  \brief  synchronize a specific device stream
     *  \param  stream_id   index of the stream to be synchronized
     *  \return POS_SUCCESS for successfully synchronizing
     */
    pos_retval_t __sync_stream(uint64_t stream_id=0) override;
    /* ===================== platform-specific functions ===================== */
};


// declarations of CUDA handles
class POSHandle_CUDA_Context;
class POSHandle_CUDA_Device;
class POSHandle_CUDA_Event;
class POSHandle_CUDA_Function;
class POSHandle_CUDA_Memory;
class POSHandle_CUDA_Module;
class POSHandle_CUDA_Stream;
class POSHandle_CUDA_Var;
class POSHandle_cuBLAS_Context;

// declarations of managers for CUDA handles
class POSHandleManager_CUDA_Context;
class POSHandleManager_CUDA_Device;
class POSHandleManager_CUDA_Event;
class POSHandleManager_CUDA_Function;
class POSHandleManager_CUDA_Memory;
class POSHandleManager_CUDA_Module;
class POSHandleManager_CUDA_Stream;
class POSHandleManager_CUDA_Var;
class POSHandleManager_cuBLAS_Context;

// definitions
#include "pos/cuda_impl/handle/context.h"
#include "pos/cuda_impl/handle/device.h"
#include "pos/cuda_impl/handle/event.h"
#include "pos/cuda_impl/handle/function.h"
#include "pos/cuda_impl/handle/memory.h"
#include "pos/cuda_impl/handle/module.h"
#include "pos/cuda_impl/handle/stream.h"
#include "pos/cuda_impl/handle/var.h"
#include "pos/cuda_impl/handle/cublas.h"
