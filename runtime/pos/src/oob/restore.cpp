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
#include <string>
#include <filesystem>

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/oob/restore.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/workspace.h"
#include "pos/include/agent.h"
#include "pos/include/command.h"
#include "pos/cuda_impl/client.h"


namespace oob_functions {

/*!
 *  \related    kPOS_OOB_Msg_CLI_Restore
 *  \brief      signal for restore the state of a specific client
 */
namespace cli_restore {
    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;
        std::string retmsg;
        POSClient *client;
        std::string ckpt_dir, client_ckpt_path;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(oob_server);

        POS_CHECK_POINTER(payload = (oob_payload_t*)msg->payload);

        // make sure the directory exist
        ckpt_dir = std::string(payload->ckpt_dir) + std::string("/phos");
        if (!std::filesystem::exists(ckpt_dir)) {
            payload->retval = POS_FAILED_NOT_EXIST;
            retmsg = std::string("no ckpt dir exist: ") + ckpt_dir.c_str();
            goto response;
        }

        // restore client in the workspace
        POS_LOG("try restore client");
        client_ckpt_path = ckpt_dir + std::string("/c.bin");
        if (!std::filesystem::exists(client_ckpt_path)) {
            payload->retval = POS_FAILED_NOT_EXIST;
            retmsg = std::string("ckpt corrupted: no client data");
            goto response;
        }
        if(unlikely(POS_SUCCESS != (payload->retval = ws->restore_client(client_ckpt_path, &client)))){
            retmsg = std::string("see posd log for more details");
            goto response;
        }
        POS_CHECK_POINTER(client);
        POS_ASSERT(client->status != kPOS_ClientStatus_Active);
        POS_LOG("restored client");

        // restore handle in the client handle manager
        if(unlikely(POS_SUCCESS != (
            payload->retval = client->restore_handles(ckpt_dir)
        ))){
            retmsg = std::string("see posd log for more details");
            goto response;
        }
        POS_LOG("restored handle");

        // reload unexecuted APIs in the client queue (async thread)
        if(unlikely(POS_SUCCESS != (
            payload->retval = client->restore_apicxts(ckpt_dir)
        ))){
            retmsg = std::string("see posd log for more details");
            goto response;
        }
        POS_LOG("restored apicxts");

        // now it's time to let client start to work
        client->status = kPOS_ClientStatus_Active;  // start polling its internal queue
        POS_LOG("resumed execution of client");

    response:
        POS_ASSERT(retmsg.size() < kServerRetMsgMaxLen);
        __POS_OOB_SEND();

        return retval;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        pos_retval_t retval = POS_SUCCESS;
        oob_call_data_t *cm;
        oob_payload_t *payload;

        msg->msg_type = kPOS_OOB_Msg_CLI_Restore;

        POS_CHECK_POINTER(call_data);
        cm = (oob_call_data_t*)call_data;

        // setup payload
        memset(msg->payload, 0, sizeof(msg->payload));
        payload = (oob_payload_t*)msg->payload;
        memcpy(payload->ckpt_dir, cm->ckpt_dir, kCkptFilePathMaxLen);

        __POS_OOB_SEND();

        // wait until the posd finished 
        __POS_OOB_RECV();
        cm->retval = payload->retval;
        memcpy(cm->retmsg, payload->retmsg, kServerRetMsgMaxLen);

    exit:
        return retval;
    }
} // namespace cli_restore

} // namespace oob_functions
