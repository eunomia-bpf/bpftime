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

#include "string.h"

#include "pos/include/common.h"

#if POS_CONF_RUNTIME_EnableTrace
    // define a new list of tracing couter
    #define POS_TRACE_REDUCER_DEF(list_name, data_type, ...)        \
        typedef struct __pos_reducer_list_##list_name {             \
            struct __data {                                         \
                data_type __VA_ARGS__;                              \
            };                                                      \
            struct __counter {                                      \
                uint64_t __VA_ARGS__;                               \
            };                                                      \
            __data reducers;                                        \
            __counter counters;                                     \
        } __pos_reducer_list_##list_name##_t;

    // declare a new list of reducers
    #define POS_TRACE_REDUCER_DECLARE(list_name)               \
        __pos_reducer_list_##list_name##_t __prl_##list_name;

    // externally declare a new list of reducers
    #define POS_TRACE_REDUCER_EXTERN_DECLARE(list_name)        \
        extern __pos_reducer_list_##list_name##_t __prl_##list_name;

    // reset all reducers and counters
    #define POS_TRACE_REDUCER_RESET(list_name)                                                          \
        memset(&(__prl_##list_name.reducers), 0, sizeof(__pos_reducer_list_##list_name##_t::__data));   \
        memset(&(__prl_##list_name.counters), 0, sizeof(__pos_reducer_list_##list_name##_t::__counter));

    // max method
    #define POS_TRACE_REDUCER_MAX(list_name, reducer_name, value)   \
        __prl_##list_name.reducers.reducer_name =                   \
            __prl_##list_name.reducers.reducer_name < value ?       \
            value :                                                 \
            __prl_##list_name.reducers.reducer_name

    // min method
    #define POS_TRACE_REDUCER_MIN(list_name, reducer_name, value)   \
        __prl_##list_name.reducers.reducer_name =                   \
            __prl_##list_name.reducers.reducer_name > value ?       \
            value :                                                 \
            __prl_##list_name.reducers.reducer_name

    // avg method
    #define POS_TRACE_REDUCER_AVG_APPEND(list_name, reducer_name, value)    \
        __prl_##list_name.counters.reducer_name += 1;                       \
        __prl_##list_name.reducers.reducer_name += value;

    // obtain max/min value
    #define POS_TRACE_REDUCER_MAXMIN_GET(list_name, reducer_name)   \
        __prl_##list_name.counters.reducer_name

    // obtain max/min value
    #define POS_TRACE_REDUCER_AVG_GET(list_name, reducer_name, data_type)                                   \
    (                                                                                                       \
        __prl_##list_name.counters.reducer_name == 0 ?                                                      \
        (data_type)(0) :                                                                                    \
        ((data_type)(__prl_##list_name.reducers.reducer_name / __prl_##list_name.counters.reducer_name))    \
    )
#else
    #define POS_TRACE_REDUCER_DEF(list_name, data_type, ...)
    #define POS_TRACE_REDUCER_DECLARE(list_name)
    #define POS_TRACE_REDUCER_EXTERN_DECLARE(list_name)
    #define POS_TRACE_REDUCER_RESET(list_name)
    #define POS_TRACE_REDUCER_MAX(list_name, reducer_name, value)
    #define POS_TRACE_REDUCER_MIN(list_name, reducer_name, value)
    #define POS_TRACE_REDUCER_AVG_APPEND(list_name, reducer_name, value)
    #define POS_TRACE_REDUCER_MAXMIN_GET(list_name, reducer_name)
    #define POS_TRACE_REDUCER_AVG_GET(list_name, reducer_name, data_type)
#endif
