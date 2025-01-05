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

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/workspace.h"
#include "pos/include/agent.h"

#include "pos/cuda_impl/client.h"

namespace oob_functions {

/*!
 *  \related    kPOS_OOB_Msg_CLI_Migration_Signal
 *  \brief      migration signal send from CRIU action script
 */
namespace cli_migration_signal {
    // payload format
    typedef struct oob_payload {
        /* client */
        uint64_t client_uuid;
        uint32_t remote_ipv4;
        uint32_t port;
        /* server */
    } oob_payload_t;

    typedef struct migration_cli_meta {
        uint64_t client_uuid;
    } migration_cli_meta_t;

    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        #if POS_CONF_EVAL_MigrOptLevel > 0
        #else
            POS_WARN("received migration signal, but POS is compiled without migration support, omit");
        #endif
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
    }
} // namespace cli_migration_signal


} // namespace oob_functions
