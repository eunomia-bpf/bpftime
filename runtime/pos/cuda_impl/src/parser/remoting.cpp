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

#include "pos/include/common.h"
#include "pos/include/handle.h"
#include "pos/include/api_context.h"

#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/api_context.h"

namespace ps_functions {

/*!
 *  \related    [Cricket Adapt] rpc_dinit
 *  \brief      disconnect of RPC connection
 */
namespace remoting_deinit {
    // parser function
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSAPIContext_QE *ckpt_wqe;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        // mark this sync call can be returned after parsing
        wqe->status = kPOS_API_Execute_Status_Return_Without_Worker;

    exit:
        return retval;
    }
}; // namespace remoting_deinit

}; // namespace ps_functions
