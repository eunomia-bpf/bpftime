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
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/oob.h"

namespace oob_functions {


namespace agent_register_client {
    static constexpr uint64_t kMaxJobNameLen =  256;

    // payload format
    typedef struct oob_payload {
        /* client */
        char job_name[kMaxJobNameLen+1];
        __pid_t pid;
        /* server */
        bool is_registered;
    } oob_payload_t;

    // metadata of the client-side call
    typedef struct oob_call_data {
        std::string job_name;
        __pid_t pid;
    } oob_call_data_t;
} // namespace agent_register_client


} // namespace oob_functions
