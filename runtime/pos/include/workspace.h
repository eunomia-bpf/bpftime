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

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <atomic>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/command.h"
#include "pos/include/handle.h"
#include "pos/include/client.h"
#include "pos/include/parser.h"
#include "pos/include/worker.h"
#include "pos/include/transport.h"
#include "pos/include/oob.h"
#include "pos/include/api_context.h"
#include "pos/include/utils/timer.h"


// forward declaration
class POSWorkspace;


/*!
 *  \brief  function prototypes for cli oob server
 */
namespace oob_functions {
    POS_OOB_DECLARE_SVR_FUNCTIONS(agent_register_client);
    POS_OOB_DECLARE_SVR_FUNCTIONS(agent_unregister_client);
    POS_OOB_DECLARE_SVR_FUNCTIONS(cli_ckpt_predump);
    POS_OOB_DECLARE_SVR_FUNCTIONS(cli_ckpt_dump);
    POS_OOB_DECLARE_SVR_FUNCTIONS(cli_restore);
    POS_OOB_DECLARE_SVR_FUNCTIONS(cli_trace_resource);
}; // namespace oob_functions


/*!
 *  \brief  runtime workspace configuration
 *  \note   these configurations can be updated via CLI or workspace internal programs
 */
class POSWorkspaceConf {
 public:
    POSWorkspaceConf(POSWorkspace *root_ws);
    ~POSWorkspaceConf() = default;

    // configuration index in this container
    enum ConfigType : uint16_t {
        kRuntimeDaemonLogPath = 0,
        kRuntimeTraceResourceEnabled,
        kRuntimeTracePerformanceEnabled,
        kRuntimeTraceDir,
        kEvalCkptIntervfalMs,
        kUnknown
    }; 

    /*!
     *  \brief  set sepecific configuration in the workspace
     *  \note   should be thread-safe
     *  \param  conf_type   type of the configuration
     *  \param  val         value to set
     *  \return POS_SUCCESS for successfully setting
     */
    pos_retval_t set(ConfigType conf_type, std::string val);

    /*!
     *  \brief  obtain sepecific configuration in the workspace
     *  \note   should be thread-safe
     *  \param  conf_type   type of the configuration
     *  \param  val         value to get
     *  \return POS_SUCCESS for successfully getting
     */
    pos_retval_t get(ConfigType conf_type, std::string& val);

 private:
    friend class POSWorkspace;

    // ====== runtime configurations ======
    // path of the daemon's log
    std::string _runtime_daemon_log_path;
    // whether the workspace is in trace mode
    bool _runtime_trace_resource;
    bool _runtime_trace_performance;
    std::string _runtime_trace_dir;

    // ====== evaluation configurations ======
    // continuous checkpoint interval (ticks)
    uint64_t _eval_ckpt_interval_ms;
    uint64_t _eval_ckpt_interval_tick;

    // workspace that this configuration container attached to
    POSWorkspace *_root_ws;

    // mutex to avoid contension
    std::mutex _mutex;
};


/*!
 * \brief   base workspace of PhoenixOS
 */
class POSWorkspace {
 public:
    /*!
     *  \brief  constructor
     */
    POSWorkspace();

    /*!
     *  \brief  deconstructor
     */
    ~POSWorkspace();

    /*!
     *  \brief  initialize the workspace
     *  \return POS_SUCCESS for successfully initialization
     */
    pos_retval_t init();

    /*!
     *  \brief  shutdown the POS server
     */
    pos_retval_t deinit();


    /* =============== client management functions =============== */
 public:
    /*!
     *  \brief  create and add a new client to the workspace
     *  \param  param   parameter to create the client
     *  \param  clnt    pointer to the POSClient to be added
     *  \return POS_SUCCESS for successfully added
     */
    pos_retval_t create_client(pos_create_client_param_t& param, POSClient** clnt);

    /*!
     *  \brief  remove a client by given uuid
     *  \param  uuid    specified uuid of the client to be removed
     *  \return POS_FAILED_NOT_EXIST for no client with the given uuid exists;
     *          POS_SUCCESS for successfully removing
     */
    pos_retval_t remove_client(pos_client_uuid_t uuid);

    /*!
     *  \brief  restore a client to the workspace, based on given ckpt file
     *  \param  ckpt_file   path to the checkpoint file of the client
     *  \param  clnt        pointer to the restored client
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t restore_client(std::string& ckpt_file, POSClient** clnt);


    /*!
     *  \brief  obtain client by given uuid
     *  \param  uuid    uuid of the client
     *  \return pointer to the corresponding POSClient
     */
    inline POSClient* get_client_by_uuid(pos_client_uuid_t uuid){
        POSClient *retval = nullptr;

        if(uuid >= this->_client_list.size()){
            goto exit;
        }
        retval = this->_client_list[uuid];
        
    exit:
        return retval;
    }


    /*!
     *  \brief  obtain client by given pid
     *  \param  pid     pid of the client
     *  \return pointer to the corresponding POSClient
     */
    POSClient* get_client_by_pid(__pid_t pid);

 protected:
    /*!
     *  \brief  create a specific-implemented client
     *  \param  parameter to create the client
     *  \param  client  pointer to the client to be created
     *  \return POS_SUCCESS for successfully creating
     */
    virtual pos_retval_t __create_client(pos_create_client_param_t& param, POSClient **client){
        return POS_FAILED_NOT_IMPLEMENTED;
    }

    /*!
     *  \brief  destory a specific-implemented client
     *  \param  client  pointer to the client to be destoried
     *  \return POS_SUCCESS for successfully destorying
     */
    virtual pos_retval_t __destory_client(POSClient *client){
        return POS_FAILED_NOT_IMPLEMENTED;
    }


    // map of clients
    std::vector<POSClient*> _client_list;
    std::map<__pid_t, POSClient*> _pid_client_map;

    // the max uuid that has been recorded
    pos_client_uuid_t _current_max_uuid;

    /* ============ end of client management functions =========== */

 public:
    /*!
     *  \brief  entrance of POS :)
     *  \param  api_id          index of the called API
     *  \param  uuid            uuid of the remote client
     *  \param  is_sync         indicate whether the api is a sync one
     *  \param  param_desps     description of all parameters of the call
     *  \param  ret_data        pointer to the data to be returned
     *  \param  ret_data_len    length of the data to be returned
     *  \return return code on specific XPU platform
     */
    int pos_process(
        uint64_t api_id, pos_client_uuid_t uuid, std::vector<POSAPIParamDesp_t> param_desps,
        void* ret_data=nullptr, uint64_t ret_data_len=0
    );

    /*!
     *  \brief  try obtain the aliveness of the client, if it isn't ready, the remoting framework should stop receiving request
     *  \param  uuid    uuid of the client
     *  \return 0 for client not ready
     *          1 for client ready
     */    
    inline int try_lock_client(pos_client_uuid_t uuid){
        volatile POSClient *client;
        int retval = 1;

        if(unlikely(this->_client_list.size() <= uuid)){
            // POS_WARN_C("try to require access to non-exist client: uuid(%lu)", uuid);
            return 0;
        }

        client = this->_client_list[uuid];
        if(unlikely(client == nullptr)){
            // POS_WARN_C("try to require access to non-exist client: uuid(%lu)", uuid);
            retval = 0; goto exit;
        }

        if(unlikely(client->offline_counter > 0)){
            // confirm to the pos worker thread
            if(client->offline_counter == 1){
                POS_DEBUG_C("confirm client offline: uuid(%lu)", uuid);
                client->offline_counter += 1;
            }
            retval = 0; goto exit;
        }

    exit:
        return retval;
    }

    // api manager
    POSApiManager *api_mgnr;

    // idx of all resources types
    std::vector<uint64_t> resource_type_idx;

    // idx of resource types whose state should be saved during predump
    std::vector<uint64_t> stateful_resource_type_idx;

    // idx of resource types whose state could be saved after predump
    std::vector<uint64_t> stateless_resource_type_idx;

    // dynamic configuration of this workspace
    POSWorkspaceConf ws_conf;

    // TSC timer of the workspace
    POSUtilTscTimer tsc_timer;

 protected:
    /*!
     *  \brief  out-of-band server
     *  \note   use cases: intereact with CLI, and also agent-side
     */
    POSOobServer *_oob_server;

    /*!
     *  \brief  initialize the workspace
     *  \note   create device context inside this function, implementation on specific platform
     *  \return POS_SUCCESS for successfully initialization
     */
    virtual pos_retval_t __init() { return POS_FAILED_NOT_IMPLEMENTED; }

    /*!
     *  \brief  deinitialize the workspace
     *  \note   destory device context inside this function, implementation on specific platform
     *  \return POS_SUCCESS for successfully deinitialization
     */
    virtual pos_retval_t __deinit(){ return POS_FAILED_NOT_IMPLEMENTED; }

    /*!
     *  \brief  preserve resource on posd
     *  \param  rid     the resource type to preserve
     *  \param  data    source data for preserving
     *  \return POS_SUCCESS for successfully preserving
     */
    virtual pos_retval_t preserve_resource(pos_resource_typeid_t rid, void *data){
        return POS_FAILED_NOT_IMPLEMENTED;
    }
    
    void parse_command_line_options(int argc, char *argv[]);
};
