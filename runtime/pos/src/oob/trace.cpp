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
#include "pos/include/oob/trace.h"
#include "pos/include/log.h"
#include "pos/include/api_context.h"
#include "pos/include/workspace.h"
#include "pos/include/agent.h"
#include "pos/include/command.h"

#include "pos/cuda_impl/client.h"


namespace oob_functions {

/*!
 *  \related    kPOS_OOB_Msg_CLI_Trace
 *  \brief      signal for marking the workspace as trace mode
 */
namespace cli_trace_resource {
    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;
        std::string retmsg;
        trace_action action;
        std::string trace_dir;

        payload = (oob_payload_t*)msg->payload;

        // make sure the directory exist
        trace_dir = std::string(payload->trace_dir);
        if (!std::filesystem::exists(trace_dir)) {
            try {
                std::filesystem::create_directories(trace_dir);
            } catch (const std::filesystem::filesystem_error& e) {
                retmsg = std::string("failed to create dir: ") + e.what();
                payload->retval = POS_FAILED;
                memcpy(payload->retmsg, retmsg.c_str(), retmsg.size());
                goto response;
            }
            POS_LOG("create resource trace dir: %s", trace_dir.c_str());
        } else {
            POS_LOG("reused existing resource trace dir: %s", trace_dir.c_str());
        }
        ws->ws_conf.set(POSWorkspaceConf::ConfigType::kRuntimeTraceDir, trace_dir);

        // switch workspace configurationn
        switch (payload->action)
        {
        case kTrace_Start:
            ws->ws_conf.set(POSWorkspaceConf::ConfigType::kRuntimeTraceResourceEnabled, "true");
            break;

        case kTrace_Stop:
            ws->ws_conf.set(POSWorkspaceConf::ConfigType::kRuntimeTraceResourceEnabled, "false");
            break;

        default:
            POS_ERROR_DETAIL("unregornized trace action: %u, this is a bug", payload->action);
        }
        payload->retval = POS_SUCCESS;

    response:
        POS_ASSERT(retmsg.size() < kServerRetMsgMaxLen);
        __POS_OOB_SEND();

    exit:
        return retval;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        pos_retval_t retval = POS_SUCCESS;
        oob_call_data_t *cm;
        oob_payload_t *payload;

        msg->msg_type = kPOS_OOB_Msg_CLI_Trace_Resource;

        POS_CHECK_POINTER(call_data);
        cm = (oob_call_data_t*)call_data;

        // setup payload
        memset(msg->payload, 0, sizeof(msg->payload));
        payload = (oob_payload_t*)msg->payload;
        payload->action = cm->action;
        memcpy(payload->trace_dir, cm->trace_dir, kTraceFilePathMaxLen);

        __POS_OOB_SEND();

        // wait until the posd finished 
        __POS_OOB_RECV();
        cm->retval = payload->retval;
        memcpy(cm->retmsg, payload->retmsg, kServerRetMsgMaxLen);

    exit:
        return retval;
    }


} // namespace cli_trace_resource

} // namespace oob_functions
