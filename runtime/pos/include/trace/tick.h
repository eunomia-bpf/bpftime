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
#include "pos/include/utils/timer.h"

#if POS_CONF_RUNTIME_EnableTrace
    // define a new list of tracing tikcs
    #define POS_TRACE_TICK_LIST_DEF(list_name, ...)                     \
        typedef struct __pos_trace_list_##list_name {                   \   
            struct __data_u64 {                                         \
                uint64_t __VA_ARGS__;                                   \
            };                                                          \
            __data_u64 s_ticks;                                         \
            __data_u64 a_ticks;                                         \
            __data_u64 times;                                           \
                                                                        \
            uint64_t _last_collect_tick;                                \
            POSUtilTscTimer *_tsc_timer;                                \
                                                                        \
            __pos_trace_list_##list_name()                              \
                :   _last_collect_tick(0), _tsc_timer(nullptr){}        \
        } __pos_trace_list_##list_name##_t;

    // declare a new list of tracing counters
    #define POS_TRACE_TICK_LIST_DECLARE(list_name)                      \
        __pos_trace_list_##list_name##_t __ptl_##list_name;
    
    // externally declare a new list of tracing counters
    #define POS_TRACE_TICK_LIST_EXTERN_DECLARE(list_name)               \
        extern __pos_trace_list_##list_name##_t __ptl_##list_name;

    // set the TSC timer of the specified profiler
    #define POS_TRACE_TICK_LIST_SET_TSC_TIMER(list_name, tsc_timer)     \
        POS_CHECK_POINTER(__ptl_##list_name._tsc_timer = tsc_timer);

    // reset all tracing counters
    #define POS_TRACE_TICK_LIST_RESET(list_name)                                                        \
        memset(&(__ptl_##list_name.s_ticks), 0, sizeof(__pos_trace_list_##list_name##_t::__data_u64));  \
        memset(&(__ptl_##list_name.a_ticks), 0, sizeof(__pos_trace_list_##list_name##_t::__data_u64));  \
        memset(&(__ptl_##list_name.times), 0, sizeof(__pos_trace_list_##list_name##_t::__data_u64));

    // record start tick to a counter
    #define POS_TRACE_TICK_START(list_name, tick_name)  \
        __ptl_##list_name.s_ticks.tick_name = POSUtilTscTimer::get_tsc();

    // record end tick to a counter
    #define POS_TRACE_TICK_END(list_name, tick_name)                                \
        __ptl_##list_name.a_ticks.tick_name                                         \   
            = POSUtilTscTimer::get_tsc() - __ptl_##list_name.s_ticks.tick_name;     \
        __ptl_##list_name.times.tick_name += 1;                                 

    // append durations to a counter
    #define POS_TRACE_TICK_APPEND(list_name, tick_name)                             \
        __ptl_##list_name.a_ticks.tick_name                                         \
            += POSUtilTscTimer::get_tsc() - __ptl_##list_name.s_ticks.tick_name;    \
        __ptl_##list_name.times.tick_name += 1;                                 

    // append durations to a counter without counting time
    #define POS_TRACE_TICK_APPEND_NO_COUNT(list_name, tick_name)                    \
        __ptl_##list_name.a_ticks.tick_name                                         \
            += POSUtilTscTimer::get_tsc() - __ptl_##list_name.s_ticks.tick_name;                             

    // count one time to a counter
    #define POS_TRACE_TICK_ADD_COUNT(list_name, tick_name) \
        __ptl_##list_name.times.tick_name += 1;                    

    // reset a counter
    #define POS_TRACE_TICK_RESET(list_name, tick_name)  \
        __ptl_##list_name.s_ticks.tick_name = 0;        \
        __ptl_##list_name.a_ticks.tick_name = 0;        \
        __ptl_##list_name.times.tick_name = 0;

    // obtain counter value in microseconds
    #define POS_TRACE_TICK_GET_MS(list_name, tick_name) \
        __ptl_##list_name._tsc_timer->tick_to_ms(__ptl_##list_name.a_ticks.tick_name)

    // obtain times of recording a counter
    #define POS_TRACE_TICK_GET_TIMES(list_name, tick_name) \
        __ptl_##list_name.times.tick_name

    // obtain average duration from a counter
    #define POS_TRACE_TICK_GET_AVG_MS(list_name, tick_name) \
        POS_TRACE_TICK_GET_MS(list_name, tick_name) / (double)(POS_TRACE_TICK_GET_TIMES(list_name, tick_name))

    // try to collect statistics by given interval
    #define POS_TRACE_TICK_TRY_COLLECT(list_name, workload)                                                                 \
        if(unlikely(                                                                                                        \
            POSUtilTscTimer::get_tsc() - __ptl_##list_name._last_collect_tick > __ptl_##list_name._collect_interval_tick    \
        )){                                                                                                                 \    
            __ptl_##list_name._last_collect_tick = POSUtilTscTimer::get_tsc();                                              \
            workload                                                                                                        \
        }
#else
    #define POS_TRACE_TICK_LIST_DEF(list_name, ...)
    #define POS_TRACE_TICK_LIST_DECLARE(list_name)
    #define POS_TRACE_TICK_LIST_EXTERN_DECLARE(list_name)
    #define POS_TRACE_TICK_LIST_SET_TSC_TIMER(list_name, tsc_timer)
    #define POS_TRACE_TICK_LIST_RESET(list_name)
    #define POS_TRACE_TICK_START(list_name, tick_name)
    #define POS_TRACE_TICK_END(list_name, tick_name)
    #define POS_TRACE_TICK_APPEND(list_name, tick_name)
    #define POS_TRACE_TICK_APPEND_NO_COUNT(list_name, tick_name)
    #define POS_TRACE_TICK_ADD_COUNT(list_name, tick_name)
    #define POS_TRACE_TICK_RESET(list_name, tick_name)
    #define POS_TRACE_TICK_GET_MS(list_name, tick_name)
    #define POS_TRACE_TICK_GET_TIMES(list_name, tick_name)
    #define POS_TRACE_TICK_GET_AVG_MS(list_name, tick_name)
    #define POS_TRACE_TICK_TRY_COLLECT(list_name, workload)
#endif // POS_CONF_RUNTIME_EnableTrace
