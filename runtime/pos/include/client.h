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
#include <map>
#include <set>
#include <string>
#include <fstream>
#include <stdint.h>
#include <assert.h>
#include "pos/include/common.h"
#include "pos/include/worker.h"
#include "pos/include/parser.h"
#include "pos/include/handle.h"
#include "pos/include/command.h"
#include "pos/include/transport.h"
#include "pos/include/api_context.h"
#include "pos/include/utils/lockfree_queue.h"
#include "pos/include/utils/timer.h"


// forward declaration
class POSWorkspace;
typedef struct POSAPIContext_QE POSAPIContext_QE_t;


/*!
 *  \brief  direction of the internal queue of POS
 */
enum pos_queue_direction_t : uint8_t {
    kPOS_QueueDirection_Rpc2Parser = 0,
    kPOS_QueueDirection_Rpc2Worker,
    kPOS_QueueDirection_Parser2Worker,
    kPOS_QueueDirection_Oob2Parser,
    kPOS_QueueDirection_ParserLocal,
    kPOS_QueueDirection_WorkerLocal
};


/*!
 *  \brief  type of the internal queue of POS
 */
enum pos_queue_type_t : uint8_t {
    kPOS_Queue_Type_WQ = 0,
    kPOS_Queue_Type_CQ,
    kPOS_QueueType_ApiCxt_WQ,
    kPOS_QueueType_ApiCxt_CQ,
    kPOS_QueueType_ApiCxt_CkptDag_WQ,
    kPOS_QueueType_ApiCxt_Trace_WQ,
    kPOS_QueueType_Cmd_WQ,
    kPOS_QueueType_Cmd_CQ
};


/*!
 *  \brief  context of the client
 */
typedef struct pos_client_cxt {
    // name of the job
    std::string job_name;

    // pid of the client-side process
    __pid_t pid;

    // kernel meta path
    std::string kernel_meta_path;
    bool is_load_kernel_from_cache;

    // checkpoint file path (if any)
    std::string checkpoint_file_path;

    // indices of handle type
    std::vector<uint64_t> resource_type_idx;

    // runtime configurations
    // whether to trace resource
    bool trace_resource;
    bool trace_performance;
} pos_client_cxt_t;
#define POS_CLIENT_CXT_HEAD pos_client_cxt cxt_base;


/*!
 *  \brief  parameters to create a client in the workspace
 */
typedef struct pos_create_client_param {
    // name of the job
    std::string job_name;

    // pid of the client-side process
    __pid_t pid;

    // id of the newly created client
    pos_client_uuid_t id;

    // identify whether we're restoring a client,
    // if it's, we won't initialize initial handles
    // inside each handle manager
    bool is_restoring;
} pos_create_client_param_t;


/*!
 * \brief   status of POS client
 */
enum pos_client_status_t : uint8_t {
    kPOS_ClientStatus_CreatePending = 0,
    kPOS_ClientStatus_Active,
    kPOS_ClientStatus_Hang,
    kPOS_ClientStatus_RemotingStop
};


/*!
 *  \brief  station to store the checkpointed data
 */
typedef struct pos_client_ckpt_station {
    std::vector<std::pair<void*, uint64_t>> __chunks;
    uint64_t byte_size;

    /*!
     *  \brief  clear this station
     */
    inline void clear(){
        uint64_t i;
        void *chunk;
        for(i=0; i<__chunks.size(); i++){
            POS_CHECK_POINTER(chunk = __chunks[i].first);
            free(chunk);
        }
        __chunks.clear();
        byte_size = 0;
    }

    /*!
     *  \brief  load value to this station
     *  \tparam T   type of the value to be load
     *  \param  val value to be load
     */
    template<typename T>
    inline void load_value(const T& val){
        void *chunk;
        POS_CHECK_POINTER(chunk = malloc(sizeof(T)));
        memcpy(chunk, &val, sizeof(T));
        __chunks.push_back(std::pair<void*,uint64_t>(chunk, sizeof(T)));
        byte_size += sizeof(T);
    }

    /*!
     *  \brief  load memory area via pointer to this station
     *  \param  area    pointer to the memory area to be loaded
     *  \param  size    size of the memory area to be loaded
     */
    inline void load_mem_area(void* area, uint64_t size){
        POS_CHECK_POINTER(area);
        __chunks.push_back(std::pair<void*,uint64_t>(area, size));
        byte_size += size;
    }

    /*!
     *  \brief  dump checkpoints to binary image file
     *  \param  file_path   path of the binary image file to be dumped
     *  \return POS_SUCCESS for successfully dumpping
     *          POS_FAILED for fail to open the file
     */
    inline pos_retval_t collapse_to_image_file(std::string& file_path){
        pos_retval_t retval = POS_SUCCESS;
        uint64_t i, chunk_size;
        void *chunk;
        std::ofstream output_file;
        
        output_file.open(file_path.c_str(), std::ios::binary);
        if(unlikely(output_file.is_open() == false)){
            POS_WARN("failed to collapse checkpoint to binary file: file_path(%s)", file_path.c_str());
            retval = POS_FAILED;
            goto exit;
        }

        for(i=0; i<__chunks.size(); i++){
            POS_CHECK_POINTER(chunk = __chunks[i].first);
            chunk_size = __chunks[i].second;
            output_file.write((const char*)(chunk), chunk_size);
        }

        output_file.flush();
        output_file.close();

    exit:
        return retval;
    }

    pos_client_ckpt_station() : byte_size(0) {}

} pos_client_ckpt_station_t;


/*!
 *  \brief  base state of a remote client
 */
class POSClient {
    /* ====================== basic ====================== */
 public:
    /*!
     *  \brief  constructor
     *  \param  id  client identifier
     *  \param  pid client pid
     *  \param  cxt context to initialize this client
     *  \param  ws  pointer to the global workspace
     */
    POSClient(pos_client_uuid_t id, __pid_t pid, pos_client_cxt_t cxt, POSWorkspace *ws);
    POSClient();
    ~POSClient(){}
    

    /*!
     *  \brief  initialize of the client
     *  \note   this part can't be in the constructor as we will invoke functions
     *          that implemented by derived class
     *  \param  is_restoring    identify whether we're restoring a client, if it's, 
     *                          we won't initialize initial handles inside each 
     *                          handle manager
     */
    void init(bool is_restoring);


    /*!
     *  \brief  deinit the client
     *  \note   this part can't be in the deconstructor as we will invoke functions
     *          that implemented by derived class
     */
    void deinit();


    /*!
     *  \brief  obtain the current pc, and update it
     *  \return the current pc
     */
    inline uint64_t get_and_move_api_inst_pc(){ _api_inst_pc++; return (_api_inst_pc-1); }


    // client identifier
    pos_client_uuid_t id;

    // pid of the client
    __pid_t pid;

    // state of this client
    volatile pos_client_status_t status;

    // parser thread handle
    POSParser *parser;

    // worker thread handle
    POSWorker *worker;

    // mark whether this client is under sync call
    volatile bool is_under_sync_call;

    // counter for mark whether a client is offline
    volatile uint8_t offline_counter;

 protected:
    friend class POSWorkspace;
    friend class POSParser;
    friend class POSWorker;

    // api instance pc
    uint64_t _api_inst_pc;

    // context to initialize this client
    pos_client_cxt_t _cxt;

    // the global workspace
    POSWorkspace *_ws;
    /* ====================== basic ====================== */


    /* =============== checkpoint / restore ============== */
 public:
    /*!
     *  \brief  persist the state of this client
     *  \param  ckpt_dir    checkpoint file path
     *  \return POS_SUCCESS for successfully persist
     */
    pos_retval_t persist(std::string& ckpt_dir);


    /*!
     *  \brief  persist handle to specific checkpoint files
     *  \note   this function is currently called by the trace system,
     *          normal checkpoint routine would persist handles with
     *          API provided by POSHandle
     *  \param  with_state  whether to persist with handle state
     *  \return POS_SUCCESS for successfully persist
     */
    virtual pos_retval_t persist_handles(bool with_state){}


    /*!
     *  \brief  restore handles into this client
     *  \param  ckpt_dir    directory of checkpoing files of handles
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t restore_handles(std::string& ckpt_dir);

    
    /*!
     *  \brief  restore unexecuted API context into this client
     *  \param  ckpt_dir    directory of checkpoing files of unexecuted API context
     *  \return POS_SUCCESS for successfully restore
     */
    pos_retval_t restore_apicxts(std::string& ckpt_dir);


 protected:
    /*!
     *  \brief  reallocate a single handle with specific type in the handle manager
     *  \note   this function is called by POSClient::restore_handles
     *  \param  ckpt_file   path to the checkpoint file of the handle
     *  \param  rid         resource type index of the handle
     *  \param  hid         index of the handle
     *  \return POS_SUCCESS for successfully restore
     */
    virtual pos_retval_t __reallocate_single_handle(const std::string& ckpt_file, pos_resource_typeid_t rid, pos_u64id_t hid){
        return POS_FAILED_NOT_IMPLEMENTED;
    }


    /*!
     *  \brief  reassign handle's parent from waitlist
     *  \param  handle  pointer to the handle to be processed
     *  \return POS_SUCCESS for successfully reassigned
     */
    virtual pos_retval_t __reassign_handle_parents(POSHandle* handle){
        return POS_FAILED_NOT_IMPLEMENTED;
    }


    /*!
     *  \brief  reload unexecuted API context from checkpoint file
     *  \note   this function is called by POSClient::restore_apicxts
     *  \param  ckpt_file   path to the checkpoint file of the handle
     *  \param  type        type of the apicxt to be restored
     *  \return POS_SUCCESS for successfully restore from checkpoint file
     */
    pos_retval_t __reload_apicxt(const std::string& ckpt_file, pos_apicxt_typeid_t type);


 private: 
    /*!
     *  \brief  station of the checkpoint data, might be dumpped to file, or transmit via network
     *          to other machine
     */
    pos_client_ckpt_station_t __ckpt_station;
    /* =============== checkpoint / restore ============== */


    /* ==================== migration ==================== */
    // TODO: to be added
    /* ==================== migration ==================== */


    /* ==================== transport ==================== */
 public:
    /*!
     *  \brief  initialization of transport utilities for migration  
     *  \return POS_SUCCESS for successfully initialization
     */
    virtual pos_retval_t init_transport(){}


 protected:
    // transport endpoint
    POSTransport</* is_server */ false> *_transport;
    /* ==================== transport ==================== */


    /* =============== asynchronous queues =============== */
 public:
    /*!
     *  \brief  push queue element to specified queue
     *  \tparam qdir    queue direction
     *  \tparam qtype   type of the queue
     *  \param  qe      queue element to be pushed
     *  \return POS_SUCCESS for successfully pushed  
     */
    template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
    pos_retval_t push_q(void *qe);

    /*!
     *  \brief  poll apicxt queue element from specified queue
     *  \tparam qdir    queue direction
     *  \tparam qtype   type of the queue
     *  \param  uuid    uuid for specifying client
     *  \param  cqes    returned queue elements
     *  \return POS_SUCCESS for successfully polling
     */
    template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
    pos_retval_t poll_q(std::vector<POSAPIContext_QE*>* qes);

    /*!
     *  \brief  poll cmd queue element from specified queue
     *  \tparam qdir    queue direction
     *  \tparam qtype   type of the queue
     *  \param  uuid    uuid for specifying client
     *  \param  cqes    returned queue elements
     *  \return POS_SUCCESS for successfully polling
     */
    template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
    pos_retval_t poll_q(std::vector<POSCommand_QE_t*>* qes);

    /*!
     *  \brief  clear all elements inside the queue
     *  \tparam qdir    queue direction
     *  \tparam qtype   type of the queue
     *  \param  uuid    uuid for specifying client
     *  \return POS_SUCCESS for successfully clear
     */
    template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
    pos_retval_t clear_q();

 protected:
    // api context queue pairs from RPC frontend to parser
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_rpc2parser_wq;
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_rpc2parser_cq;

    // api context work queue from parser to worker
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_parser2worker_wq;

    // api context work queue in worker, record during ckpt
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_workerlocal_ckptdag_wq;

    // api context work queue in parser, record during resource trace mode
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_parserlocal_trace_wq;

    // api context completion queue from worker to RPC frontend
    POSLockFreeQueue<POSAPIContext_QE_t*> *_apicxt_rpc2worker_cq;

    // command queue pairs from worker to parser
    POSLockFreeQueue<POSCommand_QE_t*> *_cmd_parser2worker_wq;
    POSLockFreeQueue<POSCommand_QE_t*> *_cmd_parser2worker_cq;

    // command queue pairs from OOB to parser
    POSLockFreeQueue<POSCommand_QE_t*> *_cmd_oob2parser_wq;
    POSLockFreeQueue<POSCommand_QE_t*> *_cmd_oob2parser_cq;

 private:
    /*!
     *  \brief  create queue group for this client
     *  \return POS_SUCCESS for successfully creation
     */
    pos_retval_t __create_qgroup();


    /*!
     *  \brief  destory queue group of this client
     *  \return POS_SUCCESS for successfully destory
     */
    pos_retval_t __destory_qgroup();
    /* =============== asynchronous queues =============== */


    /* =============== resource management =============== */
 public:
    /*!
     *  \brief  all hande managers of this client
     *  \note   key:    typeid of the resource represented by the handle
     *          value:  pointer to the corresponding hande manager
     */
    std::map<pos_resource_typeid_t, POSHandleManager<POSHandle>*> handle_managers;


    /*!
     *  \brief  instantiate handle manager for all used resources
     *  \note   the children class should replace this method to initialize their 
     *          own needed handle managers
     *  \param  is_restoring    identify whether we're restoring a client, if it's, 
     *                          we won't initialize initial handles inside each 
     *                          handle manager
     *  \return POS_SUCCESS for successfully initialization
     */
    virtual pos_retval_t init_handle_managers(bool is_restoring){}


    /*!
     *  \brief      deinit: dumping handle manager for all used resources
     *  \example    CUDA function manager should export the metadata of functions
     */
    virtual void deinit_handle_managers(){}


    /*!
     *  \brief  tear down all handles
     *  \note   
     *  \return POS_SUCCESS for successfully tear down
     */
    virtual pos_retval_t tear_down_all_handles(){ return POS_FAILED_NOT_IMPLEMENTED; }


 protected:
    /*!
     *  \brief  allocate mocked resource in the handle manager according to given type
     *  \note   this function is used during restore phrase
     *  \param  type_id specified resource type index
     *  \param  bin_ptr pointer to the binary area
     *  \return POS_SUCCESS for successfully allocated
     */
    virtual pos_retval_t __allocate_typed_resource_from_binary(pos_resource_typeid_t type_id, void* bin_ptr){
        return POS_FAILED_NOT_IMPLEMENTED;
    }


    /*!
     *  \brief  obtain all resource type indices of this client
     *  \return all resource type indices of this client
     */
    virtual std::set<pos_resource_typeid_t> __get_resource_idx(){
        return std::set<pos_resource_typeid_t>();
    }


    /*!
     *  \brief  get handle manager by given resource index
     *  \param  rid    resource index
     *  \return specified handle manager
     */
    POSHandleManager<POSHandle>* __get_handle_manager_by_resource_id(pos_resource_typeid_t rid){
        if(unlikely(this->handle_managers.count(rid) == 0)){
            POS_ERROR_C_DETAIL(
                "no handle manager with specified type registered, this is a bug: type_id(%lu)", rid
            );
        }
        return this->handle_managers[rid];
    }
    /* =============== resource management =============== */
};

#define pos_get_client_typed_hm(client, resource_id, hm_type)  \
    (hm_type*)(client->handle_managers[resource_id])
