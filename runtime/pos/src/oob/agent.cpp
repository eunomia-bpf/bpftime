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
#include <mutex>

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/oob/agent.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/workspace.h"
#include "pos/include/agent.h"

#include "pos/cuda_impl/client.h"

namespace oob_functions {

/*!
 *  \related    kPOS_OOB_Msg_Agent_Register_Client
 *  \brief      register a new client to the server
 */
namespace agent_register_client {
    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;
        pos_create_client_param_t create_param;
        POSClient *clnt;

        POS_CHECK_POINTER(remote);
        POS_CHECK_POINTER(msg);
        POS_CHECK_POINTER(ws);

        payload = (oob_payload_t*)msg->payload;
        create_param.job_name = std::string(payload->job_name);
        create_param.pid = payload->pid;

        // create client
        if(unlikely(POS_SUCCESS != (
            retval = ws->create_client(create_param, &clnt)
        ))){
            POS_WARN("failed to create client: job_name(%s)", payload->job_name);
            goto exit;
        }
        POS_DEBUG("create client: job_name(%s), uuid(%lu), pid(%d)", payload->job_name, clnt->id, create_param.pid);

        msg->client_meta.uuid = clnt->id;
        payload->is_registered = true;

        __POS_OOB_SEND();

    exit:
        return retval;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        int retval = POS_SUCCESS;
        oob_payload_t *payload;
        oob_call_data_t *call_data_;

        msg->msg_type = kPOS_OOB_Msg_Agent_Register_Client;

        POS_CHECK_POINTER(call_data);
        call_data_ = (oob_call_data_t*)call_data;
        POS_ASSERT(call_data_->job_name.size() <= kMaxJobNameLen);

        POS_DEBUG(
            "[OOB %u] try registering client to the server: job_name(%s), pid(%d)",
            kPOS_OOB_Msg_Agent_Register_Client, call_data_->job_name.c_str(), call_data_->pid
        );

        memset(msg->payload, 0, sizeof(msg->payload));
        payload = (oob_payload_t*)msg->payload;
        memcpy(payload->job_name, call_data_->job_name.c_str(), call_data_->job_name.size()+1);
        payload->pid = call_data_->pid;
        __POS_OOB_SEND();

        __POS_OOB_RECV();
        if(payload->is_registered == true){
            POS_DEBUG(
                "[OOB %u] successfully register client to the server: uuid(%lu)",
                kPOS_OOB_Msg_Agent_Register_Client, msg->client_meta.uuid
            );
        } else {
            POS_WARN("[OOB %u] failed to register client to the server", kPOS_OOB_Msg_Agent_Register_Client);
            retval = POS_FAILED;
        }

        oob_clnt->set_uuid(msg->client_meta.uuid);
        agent->set_uuid(msg->client_meta.uuid);

        return retval;
    }
} // namespace agent_register_client


namespace agent_unregister_client {
    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        POS_CHECK_POINTER(remote);
        POS_CHECK_POINTER(msg);
        POS_CHECK_POINTER(ws);

        // remove client
        ws->remove_client(msg->client_meta.uuid);

        __POS_OOB_SEND();

        return POS_SUCCESS;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        msg->msg_type = kPOS_OOB_Msg_Agent_Unregister_Client;
        __POS_OOB_SEND();
        __POS_OOB_RECV();
        return POS_SUCCESS;
    }
};

} // namespace oob_functions
