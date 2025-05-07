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

#include "pos/cli/cli.h"

pos_retval_t handle_migrate(pos_cli_options_t &clio){
    pos_retval_t retval = POS_SUCCESS;

    validate_and_cast_args(
        /* clio */ clio,
        /* rules */ {
            {
                /* meta_type */ kPOS_CliMeta_Pid,
                /* meta_name */ "pid",
                /* meta_desp */ "pid of the process to be migrated",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    clio.metas.migrate.pid = std::stoull(meta_val);
                exit:
                    return retval;
                },
                /* is_required */ true
            },
            {
                /* meta_type */ kPOS_CliMeta_Dip,
                /* meta_name */ "dip",
                /* meta_desp */ "ip of destination host",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    clio.metas.migrate.dip = inet_addr(meta_val.c_str());
                exit:
                    return retval;
                },
                /* is_required */ true
            },
            {
                /* meta_type */ kPOS_CliMeta_Dport,
                /* meta_name */ "dport",
                /* meta_desp */ "port of posd on destination host",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    clio.metas.migrate.dport = std::stoul(meta_val);
                exit:
                    return retval;
                },
                /* is_required */ false
            },
        },
        /* collapse_rule */ [](pos_cli_options_t& clio) -> pos_retval_t {
            pos_retval_t retval = POS_SUCCESS;
            return retval;
        }
    );

    // step 1: remote prepare
    

    // step 2: connect 

exit:
    return retval;
}
