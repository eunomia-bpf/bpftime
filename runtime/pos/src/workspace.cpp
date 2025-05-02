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
#include <string>
#include <atomic>
#include <filesystem>
#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/proto/handle.pb.h"
#include "pos/include/proto/client.pb.h"

#include <uuid/uuid.h>


POSWorkspaceConf::POSWorkspaceConf(POSWorkspace *root_ws){
    POS_CHECK_POINTER(this->_root_ws = root_ws);

    // runtime configurations
    this->_runtime_daemon_log_path = POS_CONF_RUNTIME_DefaultDaemonLogPath;
    this->_runtime_trace_resource = false;
    this->_runtime_trace_performance = false;

    // evaluation configurations
    this->_eval_ckpt_interval_tick = this->_root_ws->tsc_timer.ms_to_tick(
        POS_CONF_EVAL_CkptDefaultIntervalMs
    );
}


pos_retval_t POSWorkspaceConf::set(ConfigType conf_type, std::string val){
    pos_retval_t retval = POS_SUCCESS;
    std::lock_guard<std::mutex> lock(this->_mutex);
    uint64_t _tmp;

    POS_ASSERT(conf_type < ConfigType::kUnknown);

    switch (conf_type)
    {
    case kRuntimeDaemonLogPath:
        // TODO:
        break;

    case kRuntimeTraceResourceEnabled:
        if(val == "true"){
            this->_runtime_trace_resource = true;
            POS_LOG_C("set workspace resource trace mode as enabled");
        } else {
            this->_runtime_trace_resource = false;
            POS_LOG_C("set workspace resource trace mode as disabled");
        }
        break;

    case kRuntimeTracePerformanceEnabled:
        if(val == "true"){
            this->_runtime_trace_performance = true;
            POS_LOG_C("set workspace performance trace mode as enabled");
        } else {
            this->_runtime_trace_performance = false;
            POS_LOG_C("set workspace performance trace mode as disabled");
        }
        break;

    case kRuntimeTraceDir:
        this->_runtime_trace_dir = val;
        break;

    case kEvalCkptIntervfalMs:
        try {
            _tmp = std::stoull(val);
        } catch (const std::invalid_argument& e) {
            POS_WARN_C("failed to set ckpt interval: %s", e.what());
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        } catch (const std::out_of_range& e) {
            POS_WARN_C("failed to set ckpt interval: %s", e.what());
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
        this->_eval_ckpt_interval_tick = static_cast<uint64_t>(
            this->_root_ws->tsc_timer.ms_to_tick(_tmp)
        );
        this->_eval_ckpt_interval_ms = _tmp;
        break;

    default:
        POS_ERROR_C_DETAIL("unknown config type %u, this is a bug", conf_type);
        break;
    }

exit:
    return retval;
}


pos_retval_t POSWorkspaceConf::get(ConfigType conf_type, std::string& val){
    pos_retval_t retval = POS_SUCCESS;
    std::lock_guard<std::mutex> lock(this->_mutex);

    POS_ASSERT(conf_type < ConfigType::kUnknown);
    switch (conf_type)
    {
    case kRuntimeDaemonLogPath:
        val = this->_runtime_daemon_log_path;
        break;

    case kRuntimeTraceResourceEnabled:
        val = std::to_string(this->_runtime_trace_resource);
        break;

    case kRuntimeTracePerformanceEnabled:
        val = std::to_string(this->_runtime_trace_performance);
        break;

    case kRuntimeTraceDir:
        val = this->_runtime_trace_dir;
        break;

    case kEvalCkptIntervfalMs:
        val = std::to_string(this->_eval_ckpt_interval_ms);
        break;

    default:
        POS_ERROR_C_DETAIL("unknown config type %u, this is a bug", conf_type);
        break;
    }

exit:
    return retval;
}


POSWorkspace::POSWorkspace() :
    _current_max_uuid(0),
    ws_conf(this)
{
    // create out-of-band server
//    _oob_server = new POSOobServer(
//        /* ws */ this,
//        /* callback_handlers */ {
//            {   kPOS_OOB_Msg_Agent_Register_Client,     oob_functions::agent_register_client::sv    },
//            {   kPOS_OOB_Msg_Agent_Unregister_Client,   oob_functions::agent_unregister_client::sv  },
//            {   kPOS_OOB_Msg_CLI_Ckpt_PreDump,          oob_functions::cli_ckpt_predump::sv         },
//            {   kPOS_OOB_Msg_CLI_Ckpt_Dump,             oob_functions::cli_ckpt_dump::sv            },
//            {   kPOS_OOB_Msg_CLI_Restore,               oob_functions::cli_restore::sv              },
//            {   kPOS_OOB_Msg_CLI_Trace_Resource,        oob_functions::cli_trace_resource::sv       },
//        },
//        /* ip_str */ POS_OOB_SERVER_DEFAULT_IP,
//        /* port */ POS_OOB_SERVER_DEFAULT_PORT
//    );
//    POS_CHECK_POINTER(_oob_server);

    // create daemon directory
    if (!std::filesystem::exists(this->ws_conf._runtime_daemon_log_path)) {
        try {
            std::filesystem::create_directories(this->ws_conf._runtime_daemon_log_path);
        } catch (const std::filesystem::filesystem_error& e) {
            POS_ERROR_C(
                "failed to create daemon log directory at %s: %s",
                this->ws_conf._runtime_daemon_log_path.c_str(),
                e.what()
            );
        }
        POS_DEBUG_C("created daemon log directory at %s", this->ws_conf._runtime_daemon_log_path.c_str());
    } else {
        POS_DEBUG_C("reused daemon log directory at %s", this->ws_conf._runtime_daemon_log_path.c_str());
    }

    // make sure the protobuf is working
    GOOGLE_PROTOBUF_VERIFY_VERSION;

//    POS_LOG(">>>>>>>>>> PhOS Workspace <<<<<<<<<<\n%s\n", pos_banner.c_str());
//    POS_LOG("PhoenixOS workspace created, welcome!");
}


POSWorkspace::~POSWorkspace(){
    // clean protobuf
    google::protobuf::ShutdownProtobufLibrary();
};


pos_retval_t POSWorkspace::init(){
    POS_DEBUG_C("initializing POS workspace...")
    return this->__init();
}


pos_retval_t POSWorkspace::deinit(){
    pos_retval_t retval;
    uint64_t i, nb_clean_client;
    POSClient *client;

    POS_DEBUG_C("deinitializing POS workspace...")

    if(likely(_oob_server != nullptr)){
        POS_DEBUG_C("shutdowning out-of-band server...");
        delete _oob_server;
    }

    POS_DEBUG_C("cleaning all clients...");
    nb_clean_client = 0;
    for(i=0; i<this->_client_list.size(); i++){
        if(this->_client_list[i] != nullptr){
            this->remove_client(i);
            nb_clean_client += 1;
        }
    }
    POS_BACK_LINE;
    POS_DEBUG_C("cleaned clients: #clients(%lu)", nb_clean_client);

    POS_DEBUG_C("deinit platform-specific context...");
    retval = this->__deinit();
    if(likely(retval == POS_SUCCESS)){
        POS_BACK_LINE;
        POS_DEBUG_C("deinited platform-specific context");
    }
    
    return retval;
}


pos_retval_t POSWorkspace::create_client(pos_create_client_param_t& param, POSClient** clnt){
    pos_retval_t retval = POS_SUCCESS;
    uuid_t uuid;

    param.id = this->_current_max_uuid;
    this->_current_max_uuid += 1;
    param.is_restoring = false;

    // create client
    retval = this->__create_client(param, clnt);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to create platform-specific client");
        goto exit;
    }

    this->_client_list.resize(this->_current_max_uuid);
    this->_client_list[(*clnt)->id] = (*clnt);

    this->_pid_client_map[param.pid] = (*clnt);
    POS_DEBUG_C("create client: addr(%p), uuid(%lu), pid(%d)", (*clnt), (*clnt)->id, param.pid);

exit:
    return retval;
}


pos_retval_t POSWorkspace::remove_client(pos_client_uuid_t uuid){
    pos_retval_t retval = POS_SUCCESS;
    POSClient *clnt;
    typename std::map<__pid_t, POSClient*>::iterator pid_client_map_iter;

    clnt = this->get_client_by_uuid(uuid);
    if(unlikely(clnt == nullptr)){
        POS_WARN_C("try to remove an non-exist client: uuid(%lu)", uuid);
        retval = POS_FAILED_NOT_EXIST;
        goto exit;
    }

    // delete from pid map
    for(pid_client_map_iter = this->_pid_client_map.begin();
        pid_client_map_iter != this->_pid_client_map.end();
        pid_client_map_iter ++
    ){
        if(pid_client_map_iter->second == clnt){
            this->_pid_client_map.erase(pid_client_map_iter);
            break;
        }
    }

    // erase from global map
    this->_client_list[uuid] = nullptr;

    // delete client
    retval = this->__destory_client(clnt);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to destory client: uuid(%lu)", uuid);
        goto exit;
    }

    POS_DEBUG_C("removed client: uuid(%lu)", uuid);

exit:
    return retval;
}


pos_retval_t POSWorkspace::restore_client(std::string ckpt_file, POSClient** clnt){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSClient client_binary;
    std::ifstream input;
    pos_create_client_param create_param;
    pos_client_uuid_t client_uuid;
    pid_t client_pid;
    std::string client_job_name;
    POSClient *tmp_client;

    POS_CHECK_POINTER(clnt);
    
    input.open(ckpt_file, std::ios::in | std::ios::binary);
    if(!input){
        POS_WARN_C("failed to open client ckpt file");
        retval = POS_FAILED;
        goto exit;
    }

    if (!client_binary.ParseFromIstream(&input)) {
        POS_WARN_C("failed to deserialize client ckpt");
        retval = POS_FAILED;
        goto exit;
    }
    create_param.id = client_binary.uuid();
    create_param.pid = client_binary.pid();
    create_param.job_name = client_binary.job_name();
    create_param.is_restoring = true;

    // TODO: fix this logic later, reassign new client id
    tmp_client = this->get_client_by_uuid(create_param.id);
    if(unlikely(tmp_client != nullptr)){
        POS_WARN_C("confliction of client uuid, %s", POS_BUG_REPORT);
        retval = POS_FAILED;
        goto exit;
    }
    if(unlikely(this->_pid_client_map.count(create_param.pid) > 0)){
        POS_WARN_C("confliction of client pid, %s", POS_BUG_REPORT);
        retval = POS_FAILED;
        goto exit;
    }

    // create client
    retval = this->__create_client(create_param, clnt);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to create platform-specific client");
        goto exit;
    }
    POS_CHECK_POINTER(clnt);
    (*clnt)->_api_inst_pc = client_binary.api_inst_pc();

    if(unlikely(this->_client_list.size() < (*clnt)->id))
        this->_client_list.resize((*clnt)->id+1);
    this->_client_list[(*clnt)->id] = (*clnt);

    this->_pid_client_map[(*clnt)->pid] = (*clnt);
    POS_DEBUG_C("restore client: addr(%p), uuid(%lu), pid(%d)", (*clnt), (*clnt)->id, (*clnt)->pid);

exit:
    if(input.is_open()){ input.close(); }
    return retval;
}


POSClient* POSWorkspace::get_client_by_pid(__pid_t pid){
    POSClient *retval = nullptr;

    if(unlikely(this->_pid_client_map.count(pid) > 0)){
        retval = this->_pid_client_map[pid];
    }

    return retval;
}


int POSWorkspace::pos_process(
    uint64_t api_id, pos_client_uuid_t uuid, std::vector<POSAPIParamDesp_t> param_desps, void* ret_data, uint64_t ret_data_len
){
    uint64_t i;
    int retval, prev_error_code = 0;
    POSClient *client = nullptr;
    POSAPIMeta_t api_meta;
    bool has_prev_error = false;
    POSAPIContext_QE* wqe;
    std::vector<POSAPIContext_QE*> cqes;
    POSAPIContext_QE* cqe;

    // TODO: comment out this once we enable piggyback support of uuid
    //      in remoting framework
    uuid = 0;
    
    // wait until client is ready
    while(client == nullptr){
        client = this->get_client_by_uuid(uuid);
    }

    // wait until client is ready
    while(client->status != kPOS_ClientStatus_Active){}

    // check whether the metadata of the API was recorded
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(this->api_mgnr->api_metas.count(api_id) == 0)){
            POS_WARN_C_DETAIL(
                "no api metadata was recorded in the api manager: api_id(%lu)", api_id
            );
            return POS_FAILED_NOT_EXIST;
        }
    #endif // POS_CONF_RUNTIME_EnableDebugCheck

    api_meta = api_mgnr->api_metas[api_id];

    // generate new work queue element
    wqe = new POSAPIContext_QE(
        /* api_id*/ api_id,
        /* uuid */ uuid,
        /* param_desps */ param_desps,
        /* id */ client->get_and_move_api_inst_pc(),
        /* retval_data */ ret_data,
        /* retval_size */ ret_data_len,
        /* pos_client */ client
    );
    POS_CHECK_POINTER(wqe);

    /*!
     *  \brief  push to the work queue
     */
    client->push_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_WQ>(wqe);

    /*!
     *  \note   if this is a sync call, we need to block until cqe is obtained
     */
    if(unlikely(api_meta.is_sync)){
        // mark the client is under sync call, so that the worker thread will make sure it will return back results
        // event though it's under dumping
        client->is_under_sync_call = true;

        while(1){
            #if POS_CONF_RUNTIME_EnableDebugCheck
                if(unlikely(this->get_client_by_uuid(uuid) == nullptr)){
                    POS_ERROR_DETAIL("client disappear during waiting of sync call, this is a bug: uuid(%lu)", uuid);
                }
            #endif

            if(unlikely(
                POS_SUCCESS != (client->template poll_q<kPOS_QueueDirection_Rpc2Parser,kPOS_QueueType_ApiCxt_CQ>(&cqes))
            )){
                POS_ERROR_C_DETAIL("failed to poll runtime cq");
            }

            if(unlikely(
                POS_SUCCESS != (client->template poll_q<kPOS_QueueDirection_Rpc2Worker,kPOS_QueueType_ApiCxt_CQ>(&cqes))
            )){
                POS_ERROR_C_DETAIL("failed to poll worker cq");
            }

        #if POS_CONF_RUNTIME_EnableDebugCheck
            if(cqes.size() > 0){
                POS_DEBUG_C("polling completion queue, obtain %lu elements: uuid(%lu)", cqes.size(), uuid);
            }
        #endif

            for(i=0; i<cqes.size(); i++){
                POS_CHECK_POINTER(cqe = cqes[i]);

                // found the called sync api
                if(cqe->id == wqe->id){
                    // setup return code
                    retval = has_prev_error ? prev_error_code : cqe->api_cxt->return_code;

                    client->is_under_sync_call = false;

                    goto exit;
                }

                // record previous async error
                if(unlikely(
                    cqe->status == kPOS_API_Execute_Status_Parser_Failed
                    || cqe->status == kPOS_API_Execute_Status_Worker_Failed
                )){
                    has_prev_error = true;
                    prev_error_code = cqe->api_cxt->return_code;
                }
            }

            cqes.clear();
        }
    } else {
        // if this is a async call, we directly return success
        retval = api_mgnr->cast_pos_retval(POS_SUCCESS, api_meta.library_id);
    }

exit:
    return retval;
}
