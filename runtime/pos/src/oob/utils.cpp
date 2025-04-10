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
 *  \note   this is a special oob routine, client-side could call this routine
 *          to mock a API call; this routine would be used in the unit-test of
 *          POS, shouldn't be used in production
 */
namespace utils_mock_api_call {
    // the metadata that need to be provided when the client want to mock the API
    typedef struct api_call_meta {
        uint64_t api_id;
        std::vector<POSAPIParamDesp_t> param_desps;
        int ret_code;
        uint8_t ret_data[512];
        uint64_t ret_data_size;
    } api_call_meta_t;

    // one parameter inside the API parameter list
    typedef struct api_param {
        uint8_t value[16];
        uint64_t size;
    } api_param_t;

    // payload format
    typedef struct oob_payload {
    /* client */
        // index of the called API
        uint64_t api_id;
    
        // parameters of the api calls
        api_param_t api_params[16];
        uint64_t nb_params;

    /* server */
        int ret_code;
        uint8_t ret_data[512];
    } oob_payload_t;

    // server
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server){
        pos_retval_t retval = POS_SUCCESS;
        oob_payload_t *payload;
        uint64_t i;
        std::vector<POSAPIParamDesp_t> param_desps;

        POS_CHECK_POINTER(remote);
        POS_CHECK_POINTER(msg);
        POS_CHECK_POINTER(ws);

        // parse payload
        payload = (oob_payload_t*)msg->payload;
        for(i=0; i<payload->nb_params; i++){
            param_desps.push_back({ 
                .value = &(payload->api_params[i].value),
                .size = payload->api_params[i].size
            });
        }

        // call api
        payload->ret_code = ws->pos_process(
            /* api_id */ payload->api_id, 
            /* uuid */ msg->client_meta.uuid,
            /* param_desps */ param_desps,
            /* ret_data */ payload->ret_data
        );

        __POS_OOB_SEND();

        return POS_SUCCESS;
    }

    // client
    pos_retval_t clnt(
        int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void* call_data
    ){
        pos_retval_t retval = POS_SUCCESS;
        uint64_t i;
        api_call_meta_t *api_call_meta;
        oob_payload_t *payload;

        msg->msg_type = kPOS_OOB_Msg_Utils_MockAPICall;

        POS_CHECK_POINTER(call_data);
        api_call_meta = (api_call_meta_t*)call_data;

        // setup payload
        memset(msg->payload, 0, sizeof(msg->payload));
        payload = (oob_payload_t*)msg->payload;
        payload->api_id = api_call_meta->api_id;
        payload->nb_params = api_call_meta->param_desps.size();
        for(i=0; i<payload->nb_params; i++){
            POS_ASSERT(api_call_meta->param_desps[i].size <= 64);
            memcpy(
                payload->api_params[i].value,
                api_call_meta->param_desps[i].value,
                api_call_meta->param_desps[i].size
            );
            payload->api_params[i].size = api_call_meta->param_desps[i].size;
        }
        __POS_OOB_SEND();
        
        __POS_OOB_RECV();
        api_call_meta->ret_code = payload->ret_code;
        memcpy(api_call_meta->ret_data, payload->ret_data, api_call_meta->ret_data_size);

        return POS_SUCCESS;
    }
} // namespace utils_mock_api_call

} // namespace oob_functions
