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

/*!
 *  \brief  hard-coded runtime configurations
 */
// default path to store log of PhOS daemon
#define POS_CONF_RUNTIME_DefaultDaemonLogPath   "/home/yangyw/daemon"

// default path to store log of PhOS client
#define POS_CONF_RUNTIME_DefaultClientLogPath   "/home/yangyw/client"

// whether to enable runtime debug check
#define POS_CONF_RUNTIME_EnableDebugCheck       1

// whether to enable runtime API hijack check
#define POS_CONF_RUNTIME_EnableHijackApiCheck   1

// whether to enable runtime trace of statistics 
#define POS_CONF_RUNTIME_EnableTrace            1

// whether to collect runtime memory trace of statistics
#define POS_CONF_RUNTIME_EnableMemoryTrace      1