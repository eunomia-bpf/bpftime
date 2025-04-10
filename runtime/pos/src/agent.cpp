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
#include <filesystem>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/agent.h"
#include "yaml-cpp/yaml.h"


POSAgentConf::POSAgentConf(POSAgent *root_agent) : _root_agent(root_agent), _pid(0) {}


pos_retval_t POSAgentConf::load_config(std::string &&file_path){
    pos_retval_t retval = POS_SUCCESS;
    YAML::Node config;

    POS_ASSERT(file_path.size() > 0);

    // obtain pid
    this->_pid = getpid();
    POS_ASSERT(this->_pid > 0);

    if(unlikely(!std::filesystem::exists(file_path))){
        POS_WARN_C(
            "failed to load agent configuration, no file exist: file_path(%s)", file_path.c_str()
        );
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    try {
        config = YAML::LoadFile(file_path);
        
        // load job name
        if(config["job_name"]){
            this->_job_name = config["job_name"].as<std::string>();
            if(unlikely(this->_job_name.size() == 0)){
                POS_WARN_C(
                    "failed to load agent configuration, no job name provided: file_path(%s)", file_path.c_str()
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
            if(unlikely(this->_job_name.size() > oob_functions::agent_register_client::kMaxJobNameLen)){
                POS_WARN_C(
                    "failed to load agent configuration, job name too long: job_name(%s), len(%lu), max(%lu)",
                    file_path.c_str(), file_path.size()+1, oob_functions::agent_register_client::kMaxJobNameLen
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
        } else {
            POS_WARN_C(
                "failed to load agent configuration, no job name provided: file_path(%s)", file_path.c_str()
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
        
        // load log path
        if(config["log_path"]){
            this->_runtime_client_log_path = config["log_path"].as<std::string>();
        } else {
            this->_runtime_client_log_path = POS_CONF_RUNTIME_DefaultClientLogPath
                                            + std::string("/") + std::to_string(this->_pid);
        }

        // load daemon addr
        if(config["daemon_addr"]){
            this->_daemon_addr = config["daemon_addr"].as<std::string>();
        } else {
            this->_daemon_addr = "127.0.0.1";
        }
    } catch (const YAML::Exception& e) {
        POS_WARN_C("failed to parse yaml file: path(%s), error(%s)", file_path.c_str(), e.what());
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    POS_DEBUG_C("loaded config from %s", file_path.c_str());

exit:
    return retval;
}


POSAgent::POSAgent() : _agent_conf(this) {
    oob_functions::agent_register_client::oob_call_data_t call_data;

    // load configurations
    if(unlikely(POS_SUCCESS != this->_agent_conf.load_config())){
        POS_ERROR_C("failed to load agent configuration");
    }

    this->_pos_oob_client = new POSOobClient(
        /* agent */ this,
        /* req_functions */ {
            {   kPOS_OOB_Msg_Agent_Register_Client,   oob_functions::agent_register_client::clnt    },
            {   kPOS_OOB_Msg_Agent_Unregister_Client, oob_functions::agent_unregister_client::clnt  },
        },
        /* local_port */ POS_OOB_CLIENT_DEFAULT_PORT,
        /* local_ip */ "0.0.0.0",
        /* server_port */ POS_OOB_SERVER_DEFAULT_PORT,
        /* server_ip */ this->_agent_conf._daemon_addr.c_str()
    );
    POS_CHECK_POINTER(this->_pos_oob_client);

    // create daemon directory
    if (std::filesystem::exists(this->_agent_conf._runtime_client_log_path)) {
        std::filesystem::remove_all(this->_agent_conf._runtime_client_log_path);
    }
    try {
        std::filesystem::create_directories(this->_agent_conf._runtime_client_log_path);
    } catch (const std::filesystem::filesystem_error& e) {
        POS_ERROR_C(
            "failed to create client log directory at %s: %s",
            this->_agent_conf._runtime_client_log_path.c_str(), e.what()
        );
    }
    POS_DEBUG_C("created client log directory at %s", this->_agent_conf._runtime_client_log_path.c_str());

    // register client
    call_data.job_name = this->_agent_conf._job_name;
    call_data.pid = this->_agent_conf._pid;
    if(POS_SUCCESS != this->_pos_oob_client->call(kPOS_OOB_Msg_Agent_Register_Client, &call_data)){
        POS_ERROR_C_DETAIL("failed to register the client");
    }
    POS_DEBUG_C(
        "successfully register client: uuid(%lu), pid(%d), job_name(%s)",
        this->_uuid, this->_agent_conf._pid, this->_agent_conf._job_name.c_str()
    );
}


POSAgent::~POSAgent(){
    if(POS_SUCCESS != this->_pos_oob_client->call(kPOS_OOB_Msg_Agent_Unregister_Client, nullptr)){
        POS_ERROR_C_DETAIL("failed to unregister the client");
    }
    delete this->_pos_oob_client;
}


pos_retval_t POSAgent::oob_call(pos_oob_msg_typeid_t id, void* data){
    POS_CHECK_POINTER(data);
    return this->_pos_oob_client->call(id, data);
}
