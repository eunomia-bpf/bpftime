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
#include <thread>
#include <vector>
#include <map>
#include <sched.h>
#include <pthread.h>
#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/utils/lockfree_queue.h"
#include "pos/include/api_context.h"
#include "pos/include/command.h"
#include "pos/include/metrics.h"

// forward declaration
class POSClient;
class POSParser;
class POSWorkspace;


/*!
 *  \brief prototype for parser function for each API call
 */
using pos_runtime_parser_function_t = pos_retval_t(*)(POSWorkspace*, POSParser*, POSAPIContext_QE*);


/*!
 *  \brief  macro for the definition of the runtime parser functions
 */
#define POS_RT_FUNC_PARSER()                                    \
    pos_retval_t parse(POSWorkspace* ws, POSParser* parser, POSAPIContext_QE* wqe)


namespace ps_functions {
    #define POS_PS_DECLARE_FUNCTIONS(api_name) namespace api_name { POS_RT_FUNC_PARSER(); }
};  // namespace ps_functions


/*!
 *  \brief  POS Parser
 */
class POSParser {
 public:
    /*!
     *  \brief  constructor
     */
    POSParser(POSWorkspace* ws, POSClient* client);

    /*!
     *  \brief  deconstructor
     */
    ~POSParser();
    
    /*!
     *  \brief  function insertion
     *  \note   this part can't be in the constructor as we will invoke functions
     *          that implemented by derived class
     *  \return POS_SUCCESS for successfully insertion
     */
    pos_retval_t init();

    /*!
     *  \brief  raise the shutdown signal to stop the daemon
     */
    void shutdown();


    /* ==================== POSParser Metrics ==================== */
 public:
    #if POS_CONF_RUNTIME_EnableTrace
        enum metrics_reducer_type_t : uint8_t {
            KERNEL_in_memories = 0,
            KERNEL_out_memories
        };
        POSMetrics_ReducerList<metrics_reducer_type_t, uint64_t> metric_reducers;

        enum metrics_counter_type_t : uint8_t {
            KERNEL_number_of_user_kernels = 0,
            KERNEL_number_of_vendor_kernels
        };
        POSMetrics_CounterList<metrics_counter_type_t> metric_counters;
    #endif
    /* ==================== POSParser Metrics ==================== */


 protected:
    // stop flag to indicate the daemon thread to stop
    volatile bool _stop_flag;

    // the daemon thread of the runtime
    std::thread *_daemon_thread;

    // global workspace
    POSWorkspace *_ws;
    
    // the coressponding client
    POSClient *_client;

    // parser function map
    std::map<uint64_t, pos_runtime_parser_function_t> _parser_functions;
    
    /*!
     *  \brief  insertion of parse functions
     *  \return POS_SUCCESS for succefully insertion
     */
    virtual pos_retval_t init_ps_functions(){ return POS_FAILED_NOT_IMPLEMENTED; }

    /*!
     *  \brief      initialization of the runtime daemon thread
     *  \example    for CUDA, one need to call API e.g. cudaSetDevice first to setup the context for a thread
     */
    virtual pos_retval_t daemon_init(){ return POS_SUCCESS; }

 private:
    /*!
     *  \brief  processing daemon of the parser
     */
    void __daemon();

    /*!
     *  \brief  insert checkpoint op to the DAG based on certain conditions
     *  \note   aware of the macro POS_CONF_EVAL_CkptEnableIncremental
     *  \return POS_SUCCESS for successfully checkpoint insertion
     */
    pos_retval_t __checkpoint_insertion();

    /*!
     *  \brief  naive implementation of checkpoint insertion procedure
     *  \note   this implementation naively insert a checkpoint op, 
     *          without any optimization hint
     *  \param  client  the client to be checkpointed
     *  \return POS_SUCCESS for successfully checkpoint insertion
     */
    pos_retval_t __checkpoint_insertion_naive();

    /*!
     *  \brief  level-1/2 optimization of checkpoint insertion procedure
     *  \note   this implementation give hints of those memory handles that
     *          been modified (INOUT/OUT) since last checkpoint
     *  \return POS_SUCCESS for successfully checkpoint insertion
     */
    pos_retval_t __checkpoint_insertion_incremental();


    /*!
     *  \brief  process command received in the parser daemon
     *  \param  cmd the received command
     *  \return POS_SUCCESS for successfully process the command
     */
    pos_retval_t __process_cmd(POSCommand_QE_t *cmd);
};
