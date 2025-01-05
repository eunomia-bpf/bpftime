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

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/client.h"
#include "pos/include/transport.h"
#include "pos/include/parser.h"


POSParser::POSParser(POSWorkspace* ws, POSClient* client) 
    : _ws(ws), _client(client), _stop_flag(false)
{
    POS_CHECK_POINTER(ws);
    POS_CHECK_POINTER(client);

    // start daemon thread
    this->_daemon_thread = new std::thread(&POSParser::__daemon, this);
    POS_CHECK_POINTER(this->_daemon_thread);

    POS_LOG_C("parser started");
};


POSParser::~POSParser(){ 
    this->shutdown(); 
}


pos_retval_t POSParser::init(){
    pos_retval_t retval;
    if(unlikely(POS_SUCCESS != (retval = this->init_ps_functions()))){
        POS_ERROR_C_DETAIL("failed to insert functions: retval(%u)", retval);
    }
    return retval;
}


void POSParser::shutdown(){ 
    this->_stop_flag = true;
    if(this->_daemon_thread != nullptr){
        if(this->_daemon_thread->joinable()){
            this->_daemon_thread->join();
        }
        delete this->_daemon_thread;
        this->_daemon_thread = nullptr;
        POS_LOG_C("parser daemon thread shutdown");
    }

    #if POS_CONF_RUNTIME_EnableTrace
        static std::unordered_map<metrics_reducer_type_t, std::string> reducer_names = {
            { KERNEL_in_memories, "KERNEL_in_memories" },
            { KERNEL_out_memories, "KERNEL_out_memories" }
        };
        static std::unordered_map<metrics_counter_type_t, std::string> counter_names = {
            { KERNEL_number_of_user_kernels, "KERNEL_number_of_user_kernels" },
            { KERNEL_number_of_vendor_kernels, "KERNEL_number_of_vendor_kernels" }
        };

        POS_LOG(
            "[Parser Metrics]:\n%s\n%s",
            this->metric_reducers.str(reducer_names).c_str(),
            this->metric_counters.str(counter_names).c_str()
        );
    #endif
}


void POSParser::__daemon(){
    uint64_t i, api_id;
    pos_retval_t parser_retval, cmd_retval;
    POSAPIMeta_t api_meta;
    uint64_t last_ckpt_tick = 0, current_tick;
    POSAPIContext_QE* apicxt_wqe;
    std::vector<POSAPIContext_QE*> apicxt_wqes;
    POSCommand_QE_t *cmd_wqe;
    std::vector<POSCommand_QE_t*> cmd_wqes;

    if(unlikely(POS_SUCCESS != this->daemon_init())){
        POS_WARN_C("failed to init daemon, worker daemon exit");
        goto exit;
    }

    while(!this->_stop_flag){
        // if the client isn't ready, the queue might not exist, we can't do any queue operation
        if(this->_client->status != kPOS_ClientStatus_Active){ continue; }

        // step 1: digest cmd from oob work queue
        cmd_wqes.clear();
        this->_client->poll_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_WQ>(&cmd_wqes);
        for(i=0; i<cmd_wqes.size(); i++){
            POS_CHECK_POINTER(cmd_wqe = cmd_wqes[i]);
            this->__process_cmd(cmd_wqe);
        }

        // step 2: digest cmd from worker completion queue
        cmd_wqes.clear();
        this->_client->poll_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_CQ>(&cmd_wqes);
        for(i=0; i<cmd_wqes.size(); i++){
            POS_CHECK_POINTER(cmd_wqe = cmd_wqes[i]);
            this->__process_cmd(cmd_wqe);
        }

        // step 3: digest apicxt from rpc work queue
        apicxt_wqes.clear();
        this->_client->poll_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_WQ>(&apicxt_wqes);

        for(i=0; i<apicxt_wqes.size(); i++){
            POS_CHECK_POINTER(apicxt_wqe = apicxt_wqes[i]);

            api_id = apicxt_wqe->api_cxt->api_id;
            api_meta = _ws->api_mgnr->api_metas[api_id];

        #if POS_CONF_RUNTIME_EnableDebugCheck
            if(unlikely(_parser_functions.count(api_id) == 0)){
                POS_ERROR_C_DETAIL(
                    "runtime has no parser function for api %lu, need to implement", api_id
                );
            }
        #endif

            apicxt_wqe->parser_s_tick = POSUtilTscTimer::get_tsc();
            parser_retval = (*(this->_parser_functions[api_id]))(this->_ws, this, apicxt_wqe);
            apicxt_wqe->parser_e_tick = POSUtilTscTimer::get_tsc();

            // set the return code
            apicxt_wqe->api_cxt->return_code = this->_ws->api_mgnr->cast_pos_retval(
                /* pos_retval */ parser_retval, 
                /* library_id */ api_meta.library_id
            );

            if(unlikely(POS_SUCCESS != parser_retval)){
                // note:    some trash programs (e.g., inside torch) can cause parser failed (on purpose)
                //          so we ignore parser failed warning
                // POS_WARN_C(
                //     "failed to execute parser function: client_id(%lu), api_id(%lu)",
                //     apicxt_wqe->client_id, api_id
                // );
                apicxt_wqe->status = kPOS_API_Execute_Status_Parser_Failed;
                apicxt_wqe->return_tick = POSUtilTscTimer::get_tsc();
                this->_client->template push_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_CQ>(apicxt_wqe);
                continue;
            }

            /*!
             *  \note       for api in type of Delete_Resource, one can directly send
             *              response to the client right after operating on mocked resources
             *  \warning    we can't apply this rule for all Create_Resource, consider the memory
             *              situation, which is passthrough addressed
             *  TODO: delete this block, should be implement in autogen system
             */
            if(unlikely(api_meta.api_type == kPOS_API_Type_Delete_Resource)){
                POS_DEBUG_C("api(%lu) is type of Delete_Resource, set as \"Return_After_Parse\"", api_id);
                apicxt_wqe->status = kPOS_API_Execute_Status_Return_After_Parse;
            }

            /*!
             *  \note       for sync api that mark as kPOS_API_Execute_Status_Return_After_Parse,
             *              we directly return the result back to the frontend side
             */
            if(     apicxt_wqe->status == kPOS_API_Execute_Status_Return_After_Parse 
                ||  apicxt_wqe->status == kPOS_API_Execute_Status_Return_Without_Worker
            ){
                apicxt_wqe->return_tick = POSUtilTscTimer::get_tsc();
                this->_client->template push_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_CQ>(apicxt_wqe);
                apicxt_wqe->has_return = true;
            }

            // launch the wqe to parser trace queue, if in resource trace mode
            if(this->_client->_cxt.trace_resource == true){
                this->_client->template push_q<kPOS_QueueDirection_ParserLocal, kPOS_QueueType_ApiCxt_Trace_WQ>(apicxt_wqe);
            }

            // skip those APIs that doesn't need worker support
            if(apicxt_wqe->status == kPOS_API_Execute_Status_Return_Without_Worker){ continue; }

            // insert apicxt_wqe to worker queue
            this->_client->template push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_ApiCxt_WQ>(apicxt_wqe);
        }
    }

exit:
    return;
}


pos_retval_t POSParser::__process_cmd(POSCommand_QE_t *cmd){
    pos_retval_t retval = POS_SUCCESS;
    POSHandleManager<POSHandle>* hm;
    POSHandle *handle;
    uint64_t i;

    POS_CHECK_POINTER(cmd);

    switch (cmd->type)
    {
    /* ========== Ckpt WQ Command from OOB thread ========== */
    case kPOS_Command_Oob2Parser_Dump:
    case kPOS_Command_Oob2Parser_PreDump:
        #if POS_CONF_EVAL_CkptOptLevel > 0
            // collect all stateful handles at this timespot to be (pre)dumped
            for(auto &handle_id : this->_ws->stateless_resource_type_idx){
                if(cmd->target_resource_type_idx.count(handle_id) == 0){ continue; }
                POS_CHECK_POINTER(
                    hm = pos_get_client_typed_hm(this->_client, handle_id, POSHandleManager<POSHandle>)
                );
                for(i=0; i<hm->get_nb_handles(); i++){
                    POS_CHECK_POINTER(handle = hm->get_handle_by_id(i));
                    cmd->record_stateless_handles(handle);
                }
            }

            // collect all stateless handles at this timespot to be (pre)dumped
            for(auto &handle_id : this->_ws->stateful_resource_type_idx){
                if(cmd->target_resource_type_idx.count(handle_id) == 0){ continue; }
                POS_CHECK_POINTER(
                    hm = pos_get_client_typed_hm(this->_client, handle_id, POSHandleManager<POSHandle>)
                );
                for(i=0; i<hm->get_nb_handles(); i++){
                    POS_CHECK_POINTER(handle = hm->get_handle_by_id(i));
                    cmd->record_stateful_handles(handle);
                }
            }
            cmd->type = cmd->type == kPOS_Command_Oob2Parser_PreDump 
                        ? kPOS_Command_Parser2Worker_PreDump
                        : kPOS_Command_Parser2Worker_Dump;
            this->_client->template push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_WQ>(cmd);
        #else
            cmd->retval = POS_FAILED_NOT_ENABLED;
            this->_client->template push_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_CQ>(cmd);
        #endif // POS_CONF_EVAL_CkptOptLevel
        break;

    /* ========== Ckpt CQ Command from worker thread ========== */
    case kPOS_Command_Parser2Worker_PreDump:
    case kPOS_Command_Parser2Worker_Dump:
        cmd->type = cmd->type == kPOS_Command_Parser2Worker_PreDump 
                    ? kPOS_Command_Oob2Parser_PreDump
                    : kPOS_Command_Oob2Parser_Dump;
        this->_client->template push_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_CQ>(cmd);
        break;

    default:
        POS_ERROR_C_DETAIL("unknown command type %u, this is a bug", cmd->type);
    }

exit:
    return retval;
}
