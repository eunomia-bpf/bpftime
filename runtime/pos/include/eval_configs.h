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
// checkpoint optimization level
#define POS_CONF_EVAL_CkptOptLevel              1

// enable increamental checkpoint
#define POS_CONF_EVAL_CkptEnableIncremental     1

// enable pipelined checkpoint
#define POS_CONF_EVAL_CkptEnablePipeline        1

// default continuous checkpoint interval (unit in ms)
#define POS_CONF_EVAL_CkptDefaultIntervalMs     1

// migration optimization level
#define POS_CONF_EVAL_MigrOptLevel              1

// restore enable using context pool
#define POS_CONF_EVAL_RstEnableContextPool      1
