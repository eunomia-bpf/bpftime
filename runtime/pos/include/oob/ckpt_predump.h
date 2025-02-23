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
#include <set>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/oob.h"

namespace oob_functions {


namespace cli_ckpt_predump {
    static constexpr uint32_t kCkptFilePathMaxLen = 256;
    static constexpr uint32_t kServerRetMsgMaxLen = 128;
    static constexpr uint32_t kTargetMaxNum = 16;
    static constexpr uint32_t kSkipTargetMaxNum = 16;

    // payload format
    typedef struct oob_payload {
        /* client */
        __pid_t pid;
        char ckpt_dir[kCkptFilePathMaxLen];
        uint32_t nb_targets;
        uint32_t nb_skip_targets;
        pos_resource_typeid_t targets[kTargetMaxNum];
        pos_resource_typeid_t skip_targets[kSkipTargetMaxNum];
        /* server */
        pos_retval_t retval;
        char retmsg[kServerRetMsgMaxLen];
    } oob_payload_t;
    static_assert(sizeof(oob_payload_t) <= POS_OOB_MSG_MAXLEN);

    // metadata from CLI
    typedef struct oob_call_data {
        /* client */
        __pid_t pid;
        char ckpt_dir[kCkptFilePathMaxLen];
        uint32_t nb_targets;
        uint32_t nb_skip_targets;
        pos_resource_typeid_t targets[kTargetMaxNum];
        pos_resource_typeid_t skip_targets[kSkipTargetMaxNum];
        /* server */
        pos_retval_t retval;
        char retmsg[kServerRetMsgMaxLen];
    } oob_call_data_t;
} // namespace cli_ckpt_predump


} // namespace oob_functions
