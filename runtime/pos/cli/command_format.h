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

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/oob.h"

/* ========= migration related commands ========= */
/*!
 *  \brief  command for initiating migration process
 *  \param  pid     [client] process id that to be migrated
 *  \param  d_ipv4  [client] IPv4 address of the host of the destination POS service
 *  \param  retval  [server] migration return value
 */
typedef struct pos_cli_migrate {
    // client
    uint64_t pid;
    uint32_t d_ipv4;
    // server
    pos_retval_t retval;
} pos_cli_migrate_t;

/*!
 *  \brief  command for preserving GPU resources (e.g., Context, Stream, etc.)
 *  \param  pid     [client] process id that to be migrated
 *  \param  d_ipv4  [client] IPv4 address of the host of the destination POS service
 *  \param  retval  [server] migration return value
 */
typedef struct pos_cli_preserve {
    // client
    // server
} pos_cli_preserve_t;
