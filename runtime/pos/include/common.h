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
#include <memory>

#include <stdint.h>
#include <assert.h>

#ifdef __GNUC__
    #define likely(x)       __builtin_expect(!!(x), 1)
    #define unlikely(x)     __builtin_expect(!!(x), 0)
#else
    #define likely(x)       (x)
    #define unlikely(x)     (x)
#endif
#define _unused(x) ((void)(x))  // make production build happy

/*!
 *  \brief  type for return value
 */
using pos_retval_t = uint8_t;

// static constexpr pos_retval_t POS_SUCCESS = 0;
// static constexpr pos_retval_t POS_WARN_DUPLICATED = 1;
// static constexpr pos_retval_t POS_WARN_NOT_READY = 2;
// static constexpr pos_retval_t POS_WARN_ABANDONED = 3;
// static constexpr pos_retval_t POS_WARN_BLOCKED = 4;
// static constexpr pos_retval_t POS_FAILED = 5;
// static constexpr pos_retval_t POS_FAILED_NOT_EXIST = 6;
// static constexpr pos_retval_t POS_FAILED_ALREADY_EXIST = 7;
// static constexpr pos_retval_t POS_FAILED_INVALID_INPUT = 8;
// static constexpr pos_retval_t POS_FAILED_DRAIN = 9;
// static constexpr pos_retval_t POS_FAILED_NOT_READY = 10;
// static constexpr pos_retval_t POS_FAILED_TIMEOUT = 11;
// static constexpr pos_retval_t POS_FAILED_NOT_IMPLEMENTED = 12;
// static constexpr pos_retval_t POS_FAILED_NOT_ENABLED = 13;
// static constexpr pos_retval_t POS_FAILED_INCORRECT_OUTPUT = 14;
// static constexpr pos_retval_t POS_FAILED_NETWORK = 15;
// static constexpr pos_retval_t POS_FAILED_DRIVER = 16;
// static constexpr pos_retval_t POS_FAILED_OOM = 17;

enum pos_retval : pos_retval_t {
    POS_SUCCESS = 0,
    POS_WARN_DUPLICATED,
    POS_WARN_NOT_READY,
    POS_WARN_ABANDONED,
    POS_WARN_BLOCKED,
    POS_FAILED,
    POS_FAILED_NOT_EXIST,
    POS_FAILED_ALREADY_EXIST,
    POS_FAILED_INVALID_INPUT,
    POS_FAILED_DRAIN,
    POS_FAILED_NOT_READY,
    POS_FAILED_TIMEOUT,
    POS_FAILED_NOT_IMPLEMENTED,
    POS_FAILED_NOT_ENABLED,
    POS_FAILED_INCORRECT_OUTPUT,
    POS_FAILED_NETWORK,
    POS_FAILED_DRIVER,
    POS_FAILED_OOM,
};


#include "pos/include/eval_configs.h"
#include "pos/include/runtime_configs.h"


#define UINT_64_MAX (1<<64 -1)

#if POS_CONF_RUNTIME_EnableDebugCheck
    #define POS_ASSERT(x)           assert(x);
    #define POS_CHECK_POINTER(ptr)  assert((ptr) != nullptr);
#else
    #define POS_ASSERT(x)           _unused (x);
    #define POS_CHECK_POINTER(ptr)  _unused (ptr);
#endif

#define POS_STATIC_ASSERT(x)    static_assert(x);

#define KB(x)   ((size_t) (x) << 10)
#define MB(x)   ((size_t) (x) << 20)
#define GB(x)   ((size_t) (x) << 30)

/*!
 *  \brief  type for resource typeid
 */
using pos_resource_typeid_t = uint32_t;

/*!
 *  \brief  type for uniquely identify a client
 */
using pos_client_uuid_t = uint64_t;

/*!
 *  \brief  type for uniquely identify a transport
 */
using pos_transport_id_t = uint64_t;

/*!
 *  \brief  type for uniquely identify a resource
 */
using pos_u64id_t = uint64_t;

/*!
 *  \brief  edge attributes for POS DAG
 */
enum pos_edge_direction_t : uint8_t {
    kPOS_Edge_Direction_In = 0,
    kPOS_Edge_Direction_Out,
    kPOS_Edge_Direction_InOut,
    kPOS_Edge_Direction_Create,
    kPOS_Edge_Direction_Delete
};


// generated in https://patorjk.com/software/taag/#p=display&f=Big&t=PhoenixOS
const std::string pos_banner =
    std::string(" _____  _                      _       ____   _____\n") +
    std::string("|  __ \\| |                    (_)     / __ \\ / ____|\n") +
    std::string("| |__) | |__   ___   ___ _ __  ___  _| |  | | (___\n") +
    std::string("|  ___/| '_ \\ / _ \\ / _ \\ '_ \\| \\ \\/ / |  | |\\___ \\\n") +
    std::string("| |    | | | | (_) |  __/ | | | |>  <| |__| |____) |\n") +
    std::string("|_|    |_| |_|\\___/ \\___|_| |_|_/_/\\_\\\\____/|_____/");
