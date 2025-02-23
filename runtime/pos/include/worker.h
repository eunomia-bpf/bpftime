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
#include <set>
#include <sched.h>
#include <pthread.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/trace.h"
#include "pos/include/metrics.h"


// forward declaration
class POSClient;
class POSHandle;
class POSWorkspace;
typedef struct POSAPIMeta POSAPIMeta_t;
typedef struct POSAPIContext_QE POSAPIContext_QE_t;
typedef struct POSCommand_QE POSCommand_QE_t;


/*!
 *  \brief prototype for worker launch function for each API call
 */
using pos_worker_launch_function_t = pos_retval_t(*)(POSWorkspace*, POSAPIContext_QE_t*);


/*!
 *  \brief  macro for the definition of the worker launch functions
 */
#define POS_WK_FUNC_LAUNCH()                                        \
    pos_retval_t launch(POSWorkspace* ws, POSAPIContext_QE_t* wqe)

namespace wk_functions {
    #define POS_WK_DECLARE_FUNCTIONS(api_name) namespace api_name { POS_WK_FUNC_LAUNCH(); }
};  // namespace ps_functions


#if POS_CONF_EVAL_CkptOptLevel == 1

typedef struct checkpoint_sync_cxt {
    // whether currently the checkpoint is active
    volatile bool ckpt_active;

    // checkpoint cmd
    POSCommand_QE_t *cmd;

    checkpoint_sync_cxt() : ckpt_active(false), cmd(nullptr) {}
} checkpoint_sync_cxt_t;

#endif


#if POS_CONF_EVAL_CkptOptLevel == 2

/*!
 *  \brief  context of the overlapped checkpoint thread
 */
typedef struct checkpoint_async_cxt {
    // flag: checkpoint thread to notify the worker thread that the previous checkpoint has done
    volatile bool TH_actve;
    volatile bool BH_active;

    // checkpoint cmd
    POSCommand_QE_t *cmd;

    // (latest) version of each handle to be checkpointed
    std::map<POSHandle*, pos_u64id_t> checkpoint_version_map;

    // all handles persisted in async checkpoint thread
    std::set<POSHandle*> persist_handles;

    // all dirty handles since start of concurrent checkpoint
    std::set<POSHandle*> dirty_handles;
    uint64_t dirty_handle_state_size;

    //  this flag should be raise by memcpy API worker function, to avoid slow down by
    //  overlapped checkpoint process
    volatile bool membus_lock;

    // thread handle
    std::thread *thread;

    // metrics
    #if POS_CONF_RUNTIME_EnableTrace
        enum metrics_reducer_type_t : uint8_t {
            CKPT_cow_bytes_by_ckpt_thread = 0,
            CKPT_cow_bytes_by_worker_thread,
            CKPT_commit_bytes_by_ckpt_thread,
            CKPT_dirty_commit_bytes
        };
        POSMetrics_ReducerList<metrics_reducer_type_t, uint64_t> metric_reducers;
    
        enum metrics_counter_type_t : uint8_t {
            CKPT_cow_done_times_by_ckpt_thread = 0,
            CKPT_cow_block_times_by_ckpt_thread,
            CKPT_cow_done_times_by_worker_thread,
            CKPT_cow_block_times_by_worker_thread,
            CKPT_commit_times_by_ckpt_thread,
            CKPT_dirty_commit_times,
            CKPT_nb_recomputation_apis,
            CKPT_nb_unexecuted_apis,
            PERSIST_handle_times,
            PERSIST_wqe_times
        };
        POSMetrics_CounterList<metrics_counter_type_t> metric_counters;

        enum metrics_ticker_type_t : uint8_t {
            COMMON_sync = 0,
            CKPT_cow_done_ticks_by_ckpt_thread,
            CKPT_cow_block_ticks_by_ckpt_thread,
            CKPT_cow_done_ticks_by_worker_thread,
            CKPT_cow_block_ticks_by_worker_thread,
            CKPT_commit_ticks_by_ckpt_thread,
            CKPT_dirty_commit_ticks,
            PERSIST_handle_ticks,
            PERSIST_wqe_ticks
        };
        POSMetrics_TickerList<metrics_ticker_type_t> metric_tickers;

        
        /*!
         *  \brief  print the metrics of the async checkpoint context
         */
        inline void print_metrics(){
            static std::unordered_map<metrics_reducer_type_t, std::string> reducer_names = {
                { CKPT_cow_bytes_by_ckpt_thread, "CoW Bytes (by Ckpt Thread)" },
                { CKPT_cow_bytes_by_worker_thread, "CoW Bytes (by Worker Thread)" },
                { CKPT_commit_bytes_by_ckpt_thread, "Commit Bytes Bytes (by Ckpt Thread)" },
                { CKPT_dirty_commit_bytes, "Dirty Copy Bytes (by Worker Thread)" },
            };

            static std::unordered_map<metrics_counter_type_t, std::string> counter_names = {
                { CKPT_cow_done_times_by_ckpt_thread, "# Handles (Cow Done by Ckpt Thread)" },
                { CKPT_cow_block_times_by_ckpt_thread, "# Handles (Cow Block by Ckpt Thread)" },
                { CKPT_cow_done_times_by_worker_thread, "# Handles (Cow Done by Worker Thread)" },
                { CKPT_cow_block_times_by_worker_thread, "# Handles (Cow Block by Worker Thread)" },
                { CKPT_commit_times_by_ckpt_thread, "# Handles (Commit by Ckpt Thread)" },
                { CKPT_dirty_commit_times, "# Dirty-copied Handles (Commit by Worker Thread)" },
                { CKPT_nb_recomputation_apis, "# Recomputation APIs" },
                { CKPT_nb_unexecuted_apis, "# Unexecuted APIs" },
                { PERSIST_handle_times, "# Persisted Handles" },
                { PERSIST_wqe_times, "# Persisted WQEs" },
            };

            static std::unordered_map<metrics_ticker_type_t, std::string> ticker_names = {
                { COMMON_sync, "Device Synchronize" },
                { CKPT_cow_done_ticks_by_ckpt_thread, "CoW Done (by Ckpt Thread)" },
                { CKPT_cow_block_ticks_by_ckpt_thread, "CoW Block (by Ckpt Thread)" },
                { CKPT_cow_done_ticks_by_worker_thread, "CoW Done (by Worker Thread)" },
                { CKPT_cow_block_ticks_by_worker_thread, "CoW Block (by Worker Thread)" },
                { CKPT_commit_ticks_by_ckpt_thread, "Commit (by Ckpt Thread)" },
                { CKPT_dirty_commit_ticks, "Dirty Copy Commit (by Worker Thread)" },
                { PERSIST_handle_ticks, "Persist Handles" },
                { PERSIST_wqe_ticks, "Persist WQEs" },
            };

            POS_LOG(
                "[AsyncCkpt Metrics]:\n%s\n%s\n%s",
                this->metric_tickers.str(ticker_names).c_str(),
                this->metric_counters.str(counter_names).c_str(),
                this->metric_reducers.str(reducer_names).c_str()
            );
        }
    #endif

    checkpoint_async_cxt() : TH_actve(false), BH_active(false), dirty_handle_state_size(0) {}
} checkpoint_async_cxt_t;

#endif // POS_CONF_EVAL_CkptOptLevel == 2


/*!
 *  \brief  POS Worker
 */
class POSWorker {
 public:
    /*!
     *  \brief  constructor
     *  \param  ws      pointer to the global workspace that create this worker
     *  \param  client  pointer to the client which this worker thread belongs to
     */
    POSWorker(POSWorkspace* ws, POSClient* client);

    /*!
     *  \brief  deconstructor
     */
    ~POSWorker();

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

    /*!
     *  \brief  generic restore procedure
     *  \note   should be invoked within the landing function, while exeucting failed
     *  \param  ws  the global workspace
     *  \param  wqe the work QE where failure was detected
     */
    static void __restore(POSWorkspace* ws, POSAPIContext_QE_t* wqe);

    /*!
     *  \brief  generic complete procedure
     *  \note   should be invoked within the landing function, while exeucting success
     *  \param  ws  the global workspace
     *  \param  wqe the work QE where failure was detected
     */
    static void __done(POSWorkspace* ws, POSAPIContext_QE_t* wqe);

    #if POS_CONF_EVAL_CkptOptLevel == 1
        checkpoint_sync_cxt_t sync_ckpt_cxt;
    #endif

    #if POS_CONF_EVAL_CkptOptLevel == 2
        checkpoint_async_cxt_t async_ckpt_cxt;
    #endif
    
    #if POS_CONF_EVAL_MigrOptLevel > 0
        // stream for precopy
        uint64_t _migration_precopy_stream_id;
    #endif

    /*!
     *  \brief  make the specified stream synchronized
     *  \param  stream_id   index of the stream to be sync, default to be 0
     */
    virtual pos_retval_t sync(uint64_t stream_id=0){
        return POS_FAILED_NOT_IMPLEMENTED;
    }

 protected:
    // stop flag to indicate the daemon thread to stop
    volatile bool _stop_flag;

    // the daemon thread of the runtime
    std::thread *_daemon_thread;

    // global workspace
    POSWorkspace *_ws;

    // corresonding client
    POSClient *_client;

    // worker function map
    std::map<uint64_t, pos_worker_launch_function_t> _launch_functions;

    #if POS_CONF_EVAL_CkptOptLevel == 2
        // stream for overlapped memcpy while computing happens
        uint64_t _ckpt_stream_id;

        // stream for doing CoW
        uint64_t _cow_stream_id;
    #endif

    #if POS_CONF_EVAL_CkptOptLevel == 2 && POS_CONF_EVAL_CkptEnablePipeline == 1
        // stream for commiting checkpoint from device
        uint64_t _ckpt_commit_stream_id;
    #endif


    /*!
     *  \brief  insertion of worker functions
     *  \return POS_SUCCESS for succefully insertion
     */
    virtual pos_retval_t init_wk_functions(){ 
        return POS_FAILED_NOT_IMPLEMENTED; 
    }

    /*!
     *  \brief      initialization of the worker daemon thread
     *  \example    for CUDA, one need to call API e.g. cudaSetDevice first to setup the context for a thread
     */
    virtual pos_retval_t daemon_init(){
        return POS_SUCCESS; 
    }

 protected:
    /*!
     *  \brief      start an ticker on GPU
     *  \example    on CUDA platform, this API is implemented using ÇUDA event
     *  \param      stream_id   index of the gpu stream to be measured
     *  \return     POS_SUCCESS for successfully started
     */
    virtual pos_retval_t start_gpu_ticker(uint64_t stream_id=0){
        return POS_FAILED_NOT_IMPLEMENTED;
    };


    /*!
     *  \brief      stop an ticker on GPU
     *  \example    on CUDA platform, this API is implemented using ÇUDA event
     *  \note       this API should cause device synchronization
     *  \param      stream_id   index of the gpu stream to be measured
     *  \param      ticker      value of the ticker
     *  \return     POS_SUCCESS for successfully started
     */
    virtual pos_retval_t stop_gpu_ticker(uint64_t& ticker, uint64_t stream_id=0){
        return POS_FAILED_NOT_IMPLEMENTED;
    }


 private:
    /*!
     *  \brief  processing daemon of the worker
     */
    void __daemon();


    #if POS_CONF_EVAL_CkptOptLevel == 0 || POS_CONF_EVAL_CkptOptLevel == 1
        /*!
         *  \brief  worker daemon with / without SYNC checkpoint support 
         *          (checkpoint optimization level 0 and 1)
         */
        void __daemon_ckpt_sync();

        /*!
         *  \brief  checkpoint procedure, should be implemented by each platform
         *  \note   this function will be invoked by level-1 ckpt
         *  \param  cmd     the checkpoint command
         *  \return POS_SUCCESS for successfully checkpointing
         */
        pos_retval_t __checkpoint_handle_sync(POSCommand_QE_t *cmd);
    #elif POS_CONF_EVAL_CkptOptLevel == 2
        /*!
         *  \brief  worker daemon with ASYNC checkpoint support (checkpoint optimization level 2)
         */
        void __daemon_ckpt_async();

        /*!
         *  \brief  [Top-half] overlapped checkpoint procedure, should be implemented by each platform
         *  \note   this thread will be raised by level-2 ckpt
         *  \note   aware of the macro POS_CONF_EVAL_CkptEnablePipeline
         *  \note   aware of the macro POS_CKPT_ENABLE_ORCHESTRATION
         */
        void __checkpoint_TH_async_thread();

        /*!
         *  \brief  [Bottom-Half] 
         *  \return ?
         */
        pos_retval_t __checkpoint_BH_sync();
    #endif

    #if POS_CONF_EVAL_MigrOptLevel > 0
        /*!
         *  \brief  worker daemon with optimized migration support (POS)
         */
        void __daemon_migration_opt();
    #endif

    /*!
     *  \brief  process command received in the worker daemon
     *  \param  cmd the received command
     *  \return POS_SUCCESS for successfully process the command
     */
    pos_retval_t __process_cmd(POSCommand_QE_t *cmd);

    /*!
     *  \brief  check and restore all broken handles, if there's any exists
     *  \param  wqe         the op to be checked and restored
     *  \param  api_meta    metadata of the called API
     *  \return POS_SUCCESS for successfully checking and restoring
     */
    pos_retval_t __restore_broken_handles(POSAPIContext_QE_t* wqe, POSAPIMeta_t *api_meta); 

    // maximum index of processed wqe index
    uint64_t _max_wqe_id;

    // mark restoring phrase
    enum pos_worker_restore_phraseid_t : uint8_t {
        kPOS_WorkRestorePhrase_Recomputation_Init = 0,
        kPOS_WorkRestorePhrase_Recomputation,
        kPOS_WorkRestorePhrase_Unexecution,
        kPOS_WorkRestorePhrase_Normal
    };
    pos_worker_restore_phraseid_t _restoring_phrase;

    // metrics
    #if POS_CONF_RUNTIME_EnableTrace
        enum metrics_reducer_type_t : uint8_t {
            __REDUCER_BASE__= 0,
            #if POS_CONF_EVAL_CkptOptLevel <= 1
                CKPT_commit_bytes,
            #endif
            RESTORE_ondemand_reload_bytes,
        };
        POSMetrics_ReducerList<metrics_reducer_type_t, uint64_t> _metric_reducers;

        enum metrics_counter_type_t : uint8_t {
            __COUNTER_BASE__= 0,
            #if POS_CONF_EVAL_CkptOptLevel <= 1
                CKPT_commit_times,
                CKPT_nb_unexecuted_apis,
                PERSIST_handle_times,
                PERSIST_wqe_times,
            #endif
            RESTORE_nb_ondemand_reload_handles,
            RESTORE_nb_ondemand_reload_state_handles,
        };
        POSMetrics_CounterList<metrics_counter_type_t> _metric_counters;

        enum metrics_ticker_type_t : uint8_t {
            __TICKER_BASE__= 0,
            #if POS_CONF_EVAL_CkptOptLevel <= 1
                COMMON_sync,
                CKPT_commit_ticks,
                PERSIST_handle_ticks,
                PERSIST_wqe_ticks,
            #else // POS_CONF_EVAL_CkptOptLevel == 2
                RESTORE_recomputation_ticks,    // gpu ticker
                RESTORE_unexecution_ticks,      // gpu ticker
            #endif
            RESTORE_ondemand_reload_ticks,
            RESTORE_ondemand_reload_state_ticks,
        };
        POSMetrics_TickerList<metrics_ticker_type_t> _metric_tickers;
        
        enum metrics_sequence_type_t : uint8_t {
            __SEQUENCE_BASE__= 0,
            #if POS_CONF_RUNTIME_EnableMemoryTrace
                KERNEL_write_state_size,
                CKPT_cow_size,
            #endif
            // note: here could have a crazy metric to collect each kernel's duration
            RESTORE_ondemand_restore_handle_nb,
            RESTORE_ondemand_restore_handle_with_state_nb,
            RESTORE_ondemand_restore_handle_state_size,
            RESTORE_ondemand_restore_handle_duration,
            RESTORE_ondemand_restore_handle_state_duration
        };
        POSMetrics_SequenceList<metrics_sequence_type_t, uint64_t> _metric_sequences;

        /*!
         *  \brief  print the metrics of the worker
         */
        void __print_metrics();
    #endif
};
