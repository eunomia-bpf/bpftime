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

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/oob/trace.h"

#include "pos/cli/cli.h"

pos_retval_t handle_trace(pos_cli_options_t &clio){
    pos_retval_t retval = POS_SUCCESS;
    oob_functions::cli_trace_resource::oob_call_data_t call_data;

    validate_and_cast_args(
        /* clio */ clio, 
        /* rules */ {
            {
                /* meta_type */ kPOS_CliMeta_SubAction,
                /* meta_name */ "subaction",
                /* meta_desp */ "action to control the behaviour of tracing resource",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;

                    if(meta_val == "start"){
                        clio.metas.trace_resource.action = oob_functions::cli_trace_resource::kTrace_Start;
                    } else if(meta_val == "stop"){
                        clio.metas.trace_resource.action = oob_functions::cli_trace_resource::kTrace_Stop;
                    } else {
                        POS_WARN("unrecognized subaction to trace-resource: %s", meta_val.c_str());
                        retval = POS_FAILED_INVALID_INPUT;
                        goto exit;
                    }

                exit:
                    return retval;
                },
                /* is_required */ true
            },
            {
                /* meta_type */ kPOS_CliMeta_Dir,
                /* meta_name */ "dir",
                /* meta_desp */ "directory to store the trace files",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    // TODO: should we cast the file path to absolute path?
                    if(meta_val.size() >= oob_functions::cli_trace_resource::kTraceFilePathMaxLen){
                        POS_WARN(
                            "ckpt file path too long: given(%lu), expected_max(%lu)",
                            meta_val.size(),
                            oob_functions::cli_trace_resource::kTraceFilePathMaxLen
                        );
                        retval = POS_FAILED_INVALID_INPUT;
                        goto exit;
                    }
                    memset(clio.metas.trace_resource.trace_dir, 0, oob_functions::cli_trace_resource::kTraceFilePathMaxLen);
                    memcpy(clio.metas.trace_resource.trace_dir, meta_val.c_str(), meta_val.size());
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

    // send trace resource request
    call_data.action = clio.metas.trace_resource.action;
    memcpy(
        call_data.trace_dir,
        clio.metas.trace_resource.trace_dir,
        oob_functions::cli_trace_resource::kTraceFilePathMaxLen
    );

    retval = clio.local_oob_client->call(kPOS_OOB_Msg_CLI_Trace_Resource, &call_data);
    if(POS_SUCCESS != call_data.retval){
        POS_WARN("set trace mode failed, %s", call_data.retmsg);
    } else {
        POS_LOG("set trace mode done");
    }

    return retval;
}
