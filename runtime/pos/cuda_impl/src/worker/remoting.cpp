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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#include "pos/include/common.h"
#include "pos/cuda_impl/worker.h"


namespace wk_functions {

/*!
 *  \related    [Cricket Adapt] rpc_dinit
 *  \brief      disconnect of RPC connection
 */
namespace remoting_deinit {
    // execution function
    POS_WK_FUNC_LAUNCH(){
        POSWorker::__done(ws, wqe);
    exit:
        return POS_SUCCESS;
    }
} // namespace remoting_deinit

} // namespace wk_functions
