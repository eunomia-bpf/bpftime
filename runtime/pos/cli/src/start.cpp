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
#include <thread>
#include <future>

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/utils/command_caller.h"

#include "pos/cli/cli.h"


pos_retval_t handle_start(pos_cli_options_t &clio){
    pos_retval_t retval = POS_SUCCESS, posd_retval;
    std::string phosd_cmd, phosd_result;

    validate_and_cast_args(
        /* clio */ clio,
        /* rules */ {
            {
                /* meta_type */ kPOS_CliMeta_Target,
                /* meta_name */ "target",
                /* meta_desp */ "target to start",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    if(meta_val != std::string("daemon")){
                        POS_WARN("unknown target %s", meta_val.c_str());
                        retval = POS_FAILED_INVALID_INPUT;
                        goto exit;
                    }
                    memcpy(clio.metas.start.target_name, meta_val.c_str(), meta_val.size() + 1);
                exit:
                    return retval;
                },
                /* is_required */ true
            }
        },
        /* collapse_rule */ [](pos_cli_options_t& clio) -> pos_retval_t {
            pos_retval_t retval = POS_SUCCESS;
            return retval;
        }
    );


    if(!strcmp(clio.metas.start.target_name, "daemon")){
        // start PhOS daemomn
        phosd_cmd = std::string("cricket-rpc-server");
        retval = POSUtil_Command_Caller::exec_sync(
            phosd_cmd,
            phosd_result,
            /* ignore_error */ false,
            /* print_stdout */ true,
            /* print_stderr */ true
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("phosd start failed");
            goto exit;
        }
    }

exit:
    return retval;
}
