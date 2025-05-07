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
#include <map>
#include <cstring>
#include <string>
#include <vector>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/oob.h"
#include "pos/include/oob/ckpt_predump.h"
#include "pos/include/oob/ckpt_dump.h"
#include "pos/include/oob/trace.h"


/*!
 *  \brief  type of the command
 */
enum pos_cli_arg : int {
    kPOS_CliAction_Unknown = 0,
    /* ============ actions ============ */
    kPOS_CliAction_Help,
    kPOS_CliAction_Start,
    kPOS_CliAction_PreDump,
    kPOS_CliAction_Dump,
    kPOS_CliAction_Restore,
    kPOS_CliAction_PreRestore,
    kPOS_CliAction_Clean,
    kPOS_CliAction_TraceResource,
    kPOS_CliAction_Migrate,
    kPOS_CliAction_PLACEHOLDER,

    /* ==== metadatas (with params) === */
    kPOS_CliMeta_SubAction,
    kPOS_CliMeta_Target,
    kPOS_CliMeta_SkipTarget,
    kPOS_CliMeta_Option,
    kPOS_CliMeta_Pid,
    kPOS_CliMeta_Dir,
    kPOS_CliMeta_Dip,
    kPOS_CliMeta_Dport,
    kPOS_CliMeta_KernelMeta,
    kPOS_CliMeta_PLACEHOLDER
};

typedef pos_cli_arg     pos_cli_action;
typedef pos_cli_arg     pos_cli_meta;

/*!
 *  \brief  convert action name from action type 
 */
static std::string pos_cli_action_name(pos_cli_arg action_type){
    switch (action_type)
    {
    case kPOS_CliAction_Help:
        return "help";

    case kPOS_CliAction_Start:
        return "start";

    case kPOS_CliAction_PreDump:
        return "pre-dump";

    case kPOS_CliAction_Dump:
        return "dump";

    case kPOS_CliAction_Restore:
        return "restore";

    case kPOS_CliAction_PreRestore:
        return "pre-restore";

    case kPOS_CliAction_Clean:
        return "clean";

    case kPOS_CliAction_TraceResource:
        return "trace-resource";

    case kPOS_CliAction_Migrate:
        return "migrate";

    default:
        return "unknown";
    }
}

typedef struct pos_cli_ckpt_metas {
    uint64_t pid;
    char ckpt_dir[oob_functions::cli_ckpt_predump::kCkptFilePathMaxLen];
    uint32_t nb_targets;
    pos_resource_typeid_t targets[oob_functions::cli_ckpt_predump::kTargetMaxNum];
    uint32_t nb_skip_targets;
    pos_resource_typeid_t skip_targets[oob_functions::cli_ckpt_predump::kSkipTargetMaxNum];
    bool do_cow;        // this option is only for dump
    bool force_recompute;  // this option is only for dump
    POS_STATIC_ASSERT(oob_functions::cli_ckpt_predump::kTargetMaxNum == oob_functions::cli_ckpt_dump::kTargetMaxNum);
    POS_STATIC_ASSERT(oob_functions::cli_ckpt_predump::kSkipTargetMaxNum == oob_functions::cli_ckpt_dump::kSkipTargetMaxNum);
} pos_cli_ckpt_metas_t;

typedef struct pos_cli_start_metas {
    char target_name[512];
} pos_cli_start_metas_t;

typedef struct pos_cli_trace_resource_metas {
    oob_functions::cli_trace_resource::trace_action action;
    char trace_dir[oob_functions::cli_trace_resource::kTraceFilePathMaxLen];
} pos_cli_trace_resource_metas_t;


typedef struct pos_cli_migrate_metas {
    uint64_t pid;
    in_addr_t dip;
    uint32_t dport;
} pos_cli_migrate_metas_t;


/*!
 *  \brief  descriptor of command line options
 */
typedef struct pos_cli_options {
    // type of the command
    pos_cli_action action_type;

    // raw option map
    std::map<pos_cli_meta, std::string> _raw_metas;
    inline void record_raw(pos_cli_meta key, std::string value){
        _raw_metas[key] = value;
    }

    POSOobClient *local_oob_client;
    POSOobClient *remote_oob_client;

    // metadata of corresponding cli option
    union {
        pos_cli_ckpt_metas_t ckpt;
        pos_cli_migrate_metas_t migrate;
        pos_cli_trace_resource_metas_t trace_resource;
        pos_cli_start_metas_t start;
    } metas;

    pos_cli_options() : local_oob_client(nullptr), remote_oob_client(nullptr), action_type(kPOS_CliAction_Unknown) {
        std::memset(&this->metas, 0, sizeof(this->metas));
    }
} pos_cli_options_t;


/*!
 *  \brief  checking rule for verifying CLI argument
 */
typedef struct pos_cli_meta_check_rule {
    pos_cli_meta meta_type;
    std::string meta_name;
    std::string meta_desp;
    using cast_func_t = pos_retval_t(*)(pos_cli_options_t&, std::string&);
    cast_func_t cast_func;
    bool is_required;
} pos_arg_check_rule_t;


/*!
 *  \brief  check rule for verifying across different CLI arguments
 */
using pos_args_collapse_rule = pos_retval_t(*)(pos_cli_options_t&);


/*!
 *  \brief  validate correctness of arguments
 *  \param  clio    all cli infomations
 *  \param  rules   checking rules
 */
static void validate_and_cast_args(pos_cli_options_t &clio, std::vector<pos_arg_check_rule_t>&& rules, pos_args_collapse_rule&& collapse_rule){
    for(auto& rule : rules){
        if(clio._raw_metas.count(rule.meta_type) == 0){
            if(rule.is_required){
                POS_ERROR(
                    "%s action requires option '%s'(%s)",
                    pos_cli_action_name(clio.action_type).c_str(),
                    rule.meta_name.c_str(),
                    rule.meta_desp.c_str()
                );
            } else {
                continue;
            }
        }

        if(unlikely(POS_SUCCESS != rule.cast_func(clio, clio._raw_metas[rule.meta_type]))){
            POS_ERROR("invalid format for '%s' option", rule.meta_name.c_str());
        }
    }

    if(unlikely(POS_SUCCESS != collapse_rule(clio))){
        POS_ERROR("failed to execute command, invalid argument provided!");
    }
}


pos_retval_t handle_help(pos_cli_options_t &clio);
pos_retval_t handle_predump(pos_cli_options_t &clio);
pos_retval_t handle_dump(pos_cli_options_t &clio);
pos_retval_t handle_migrate(pos_cli_options_t &clio);
pos_retval_t handle_trace(pos_cli_options_t &clio);
pos_retval_t handle_restore(pos_cli_options_t &clio);
pos_retval_t handle_start(pos_cli_options_t &clio);
