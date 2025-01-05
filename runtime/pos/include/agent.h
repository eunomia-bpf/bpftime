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
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/oob/agent.h"
#include "pos/include/api_context.h"


// forward declaration
class POSAgent;


/*!
 *  \brief  function prototypes for cli oob client
 */
namespace oob_functions {
    POS_OOB_DECLARE_CLNT_FUNCTIONS(agent_register_client);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(agent_unregister_client);
}; // namespace oob_functions


/*!
 *  \brief  agent configuration
 *  \note   these configurations are fixed and loaded when the client is runned
 */
class POSAgentConf {
    /*!
     *  \brief  constructor
     *  \param  agent   the agent which this configuration attached to
     */
    POSAgentConf(POSAgent *root_agent);
    ~POSAgentConf() = default;

    /*!
     *  \brief  load client config from yaml file
     *  \param  file_path   path to the configuration yaml file
     *  \return POS_SUCCESS for successfully loading
     */
    pos_retval_t load_config(std::string &&file_path = "./pos.yaml");

 private:
    friend class POSAgent;

    // the agent which thsi configuration attached to
    POSAgent *_root_agent;

    // ip addrsss of the pos daemon, commonly 127.0.0.1
    std::string _daemon_addr;

    // path of the client's log
    std::string _runtime_client_log_path;

    // name of the job
    std::string _job_name;

    // pid of the current process
    __pid_t _pid;
};


/*!
 *  \brief  client-side PhoenixOS agent, manages all POS resources
 */
class POSAgent {
 public:
    /*!
     *  \brief  constructor
     */
    POSAgent();

    /*!
     *  \brief  deconstructor
     */
    ~POSAgent();

    /*!
     *  \brief  call the out-of-band function
     *  \param  id      the out-of-band function id
     *  \param  data    payload to call the function
     *  \return according to different function definitions
     */
    pos_retval_t oob_call(pos_oob_msg_typeid_t id, void* data);

    /*!
     *  \brief  PhoenixOS API call proxy
     *  \param  api_id  index of the called API
     *  \param  params  list of parameters of the called API
     */
    pos_retval_t api_call(uint64_t api_id, std::vector<POSAPIParamDesp_t> params);

    /*!
     *  \brief  set the uuid of the client
     *  \note   this function is invoked during the registeration process 
     *          (i.e., agent_register_client oob type)
     */
    inline void set_uuid(pos_client_uuid_t id){ _uuid = id; }

 private:
    // pointer to the out-of-band client
    POSOobClient *_pos_oob_client;

    // uuid of the client
    pos_client_uuid_t _uuid;

    // configuration of the agent
    POSAgentConf _agent_conf;
};

extern POSAgent *pos_agent;
