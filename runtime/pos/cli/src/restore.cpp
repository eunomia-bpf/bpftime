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
#include <filesystem>

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/utils/command_caller.h"
#include "pos/include/oob.h"
#include "pos/include/oob/restore.h"

#include "pos/cli/cli.h"


pos_retval_t handle_restore(pos_cli_options_t &clio){
    pos_retval_t retval = POS_SUCCESS, criu_retval;
    oob_functions::cli_restore::oob_call_data_t call_data;
    std::string criu_cmd, criu_result;
    std::thread criu_thread;
    std::promise<pos_retval_t> criu_thread_promise;
    std::future<pos_retval_t> criu_thread_future = criu_thread_promise.get_future();

    validate_and_cast_args(
        /* clio */ clio,
        /* rules */ {
            {
                /* meta_type */ kPOS_CliMeta_Dir,
                /* meta_name */ "dir",
                /* meta_desp */ "directory that stores the checkpoint files",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    std::filesystem::path absolute_path;
                    
                    absolute_path = std::filesystem::absolute(meta_val);

                    if(absolute_path.string().size() >= oob_functions::cli_ckpt_dump::kCkptFilePathMaxLen){
                        POS_WARN(
                            "ckpt file path too long: given(%lu), expected_max(%lu)",
                            absolute_path.string().size(),
                            oob_functions::cli_ckpt_dump::kCkptFilePathMaxLen
                        );
                        retval = POS_FAILED_INVALID_INPUT;
                        goto exit;
                    }

                    memset(clio.metas.ckpt.ckpt_dir, 0, oob_functions::cli_restore::kCkptFilePathMaxLen);
                    memcpy(clio.metas.ckpt.ckpt_dir, absolute_path.string().c_str(), absolute_path.string().size());

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

    // send restore request to posd
    memcpy(
        call_data.ckpt_dir,
        clio.metas.ckpt.ckpt_dir,
        oob_functions::cli_restore::kCkptFilePathMaxLen
    );
    retval = clio.local_oob_client->call(kPOS_OOB_Msg_CLI_Restore, &call_data);

    // check gpu restore
    if(POS_SUCCESS != call_data.retval){
        POS_WARN("gpu restore failed, %s", call_data.retmsg);
        goto exit;
    }

    // call criu
    criu_cmd = std::string("criu restore")
                +   std::string(" -D ") + std::string(clio.metas.ckpt.ckpt_dir)
                +   std::string(" -j --display-stats");
    retval = POSUtil_Command_Caller::exec_async(
        criu_cmd, criu_thread, criu_thread_promise, criu_result,
        /* ignore_error */ false,
        /* print_stdout */ true,
        /* print_stderr */ true
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN("failed to execute CRIU");
        goto exit;
    }

    // check cpu restore
    if(criu_thread.joinable()){ criu_thread.join(); }
    criu_retval = criu_thread_future.get();
    if(POS_SUCCESS != call_data.retval){
        POS_WARN("cpu restore failed");
        goto exit;
    }

    POS_LOG("restore done");

exit:
    return retval;
}
