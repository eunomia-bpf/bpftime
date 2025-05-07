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

#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/cli/cli.h"


#define CLIENT_IP "0.0.0.0"


inline void __readin_raw_cli(int argc, char *argv[], pos_cli_options_t &clio){
    int opt_val;
    int option_index = 0;
    struct option *opt;
    char short_opt[1024] = { 0 };

    auto __get_opt_by_val = [](struct option* opt_list, int len, int val, struct option** opt){
        int i;
        struct option tmp_opt;

        POS_CHECK_POINTER(opt);
        POS_CHECK_POINTER(opt_list);

        for(i=0; i<len; i++){
            tmp_opt = opt_list[i];
            if(tmp_opt.val == val){
                (*opt) = &tmp_opt;
                goto exit;
            }
        }

        (*opt) = nullptr;

    exit:
        ;
    };

    sprintf(
        short_opt,
        /* action */    "%d%d%d%d%d%d%d%d%d"
        /* meta */      "%d:%d:%d:%d:%d:%d:%d:",
        kPOS_CliAction_Help,
        kPOS_CliAction_Start,
        kPOS_CliAction_PreDump,
        kPOS_CliAction_Dump,
        kPOS_CliAction_Restore,
        kPOS_CliAction_PreRestore,
        kPOS_CliAction_Clean,
        kPOS_CliAction_Migrate,
        kPOS_CliAction_TraceResource,
        kPOS_CliMeta_Target,
        kPOS_CliMeta_SkipTarget,
        kPOS_CliMeta_SubAction,
        kPOS_CliMeta_Pid,
        kPOS_CliMeta_Dir,
        kPOS_CliMeta_Dip,
        kPOS_CliMeta_Dport
    );

    struct option long_opt[] = {
        // action types
        {"help",            no_argument,        NULL,   kPOS_CliAction_Help},
        {"start",           no_argument,        NULL,   kPOS_CliAction_Start},
        {"pre-dump",        no_argument,        NULL,   kPOS_CliAction_PreDump},
        {"dump",            no_argument,        NULL,   kPOS_CliAction_Dump},
        {"restore",         no_argument,        NULL,   kPOS_CliAction_Restore},
        {"pre-restore",     no_argument,        NULL,   kPOS_CliAction_PreRestore},
        {"clean",           no_argument,        NULL,   kPOS_CliAction_Clean},
        {"migrate",         no_argument,        NULL,   kPOS_CliAction_Migrate},
        {"trace-resource",  no_argument,        NULL,   kPOS_CliAction_TraceResource},

        // metadatas (with param)
        {"target",      required_argument,  NULL,   kPOS_CliMeta_Target},
        {"skip-target", required_argument,  NULL,   kPOS_CliMeta_SkipTarget},
        {"subaction",   required_argument,  NULL,   kPOS_CliMeta_SubAction},
        {"option",      required_argument,  NULL,   kPOS_CliMeta_Option},
        {"pid",         required_argument,  NULL,   kPOS_CliMeta_Pid},
        {"dir",         required_argument,  NULL,   kPOS_CliMeta_Dir},
        {"dip",         required_argument,  NULL,   kPOS_CliMeta_Dip},
        {"dport",       required_argument,  NULL,   kPOS_CliMeta_Dport},

        {NULL,          0,                  NULL,   0}
    };

    while ((opt_val = getopt_long(argc, argv, (const char*)(short_opt), long_opt, &option_index)) != -1) {
        if (opt_val < kPOS_CliAction_PLACEHOLDER) {
            clio.action_type = static_cast<pos_cli_action>(opt_val);
        } else if (opt_val < kPOS_CliMeta_PLACEHOLDER) {
            __get_opt_by_val(long_opt, sizeof(long_opt), opt_val, &opt);
            POS_CHECK_POINTER(opt);
            if(opt->has_arg == true)
                clio.record_raw(static_cast<pos_cli_meta>(opt_val), optarg);
            else
                clio.record_raw(static_cast<pos_cli_meta>(opt_val), "");
        }
    }
}


inline pos_retval_t __dispatch(pos_cli_options_t &clio){
    switch (clio.action_type)
    {
    case kPOS_CliAction_Help:
        return handle_help(clio);

    case kPOS_CliAction_PreDump:
        return handle_predump(clio);

    case kPOS_CliAction_Dump:
        return handle_dump(clio);

    case kPOS_CliAction_Restore:
        return handle_restore(clio);

    case kPOS_CliAction_Migrate:
        return handle_migrate(clio);

    case kPOS_CliAction_TraceResource:
        return handle_trace(clio);

    case kPOS_CliAction_Start:
        return handle_start(clio);

    default:
        return POS_FAILED_NOT_IMPLEMENTED;
    }
}


/*!
 *  \brief  function prototypes for cli oob client
 */
namespace oob_functions {
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_predump);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_ckpt_dump);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_restore);
    POS_OOB_DECLARE_CLNT_FUNCTIONS(cli_trace_resource);
}; // namespace oob_functions


int main(int argc, char *argv[]){
    pos_retval_t retval;
    pos_cli_options_t clio;

    __readin_raw_cli(argc, argv, clio);

    clio.local_oob_client = new POSOobClient(
        /* req_functions */ {
            {   kPOS_OOB_Msg_CLI_Ckpt_PreDump,      oob_functions::cli_ckpt_predump::clnt       },
            {   kPOS_OOB_Msg_CLI_Ckpt_Dump,         oob_functions::cli_ckpt_dump::clnt          },
            {   kPOS_OOB_Msg_CLI_Restore,           oob_functions::cli_restore::clnt            },
            {   kPOS_OOB_Msg_CLI_Trace_Resource,    oob_functions::cli_trace_resource::clnt     },
        },
        /* local_port */ 10086,
        /* local_ip */ CLIENT_IP
    );
    POS_CHECK_POINTER(clio.local_oob_client);

    retval = __dispatch(clio);
    switch (retval)
    {
    case POS_SUCCESS:
        return 0;

    case POS_FAILED_NOT_IMPLEMENTED:
        POS_ERROR("unspecified action, use '-h' to check usage");

    default:
        POS_ERROR("CLI executed failed");
    }
}
