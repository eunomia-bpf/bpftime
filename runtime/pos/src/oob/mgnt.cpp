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
 *  \related    kPOS_OOB_Msg_Mgnt_OpenSession
 *  \brief      open a new OOB session
 */
namespace mgnt_open_session {
    // payload format
    typedef struct oob_payload {
        /* client */
        /* server */
        uint16_t server_port = 0;
        pos_retval_t open_result;
    } oob_payload_t;
    static_assert(sizeof(oob_payload_t) <= POS_OOB_MSG_MAXLEN);

    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;
        POSOobSession_t *new_session = nullptr;

        POS_CHECK_POINTER(payload = (oob_payload_t*)msg->payload);

        retval = oob_server->create_new_session</* is_main_session */ false>(&new_session);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed to create a new OOB session: retval(%u)", retval);
        } else {
            POS_CHECK_POINTER(new_session);
            payload->server_port = new_session->server_port;
        }
        payload->open_result = retval;

        __POS_OOB_SEND();

    exit:
        return retval;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;

        POS_CHECK_POINTER(call_data);

        // send request
        msg->msg_type = kPOS_OOB_Msg_Mgnt_OpenSession;
        memset(msg->payload, 0, sizeof(msg->payload));
        __POS_OOB_SEND();

        // register session to the client
        __POS_OOB_RECV();
        payload = (oob_payload_t*)msg->payload;
        retval = payload->open_result;
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed to request a new OOB session from server side: retval(%u)", retval);
            goto exit;
        } else {
            *((uint16_t*)(call_data)) = payload->server_port;
        }
        
    exit:
        return retval;
    }
} // namespace mgnt_open_session

} // namespace oob_functions
