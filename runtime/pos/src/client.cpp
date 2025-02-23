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
#include <algorithm>
#include <filesystem>
#include <stdint.h>
#include <assert.h>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/handle.h"
#include "pos/include/client.h"
#include "pos/include/api_context.h"
#include "pos/include/proto/client.pb.h"
#include "pos/include/proto/apicxt.pb.h"


POSClient::POSClient(pos_client_uuid_t id, __pid_t pid, pos_client_cxt_t cxt, POSWorkspace *ws) 
    :   id(id),
        pid(pid),
        status(kPOS_ClientStatus_CreatePending),
        is_under_sync_call(false),
        offline_counter(0),
        _api_inst_pc(0), 
        _cxt(cxt),
        _ws(ws)
{}


POSClient::POSClient() 
    :   id(0),
        pid(0),
        status(kPOS_ClientStatus_CreatePending),
        is_under_sync_call(false),
        offline_counter(0),
        _ws(nullptr)
{
    POS_ERROR_C("shouldn't call, just for passing compilation");
}


void POSClient::init(bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    std::map<pos_u64id_t, POSAPIContext_QE_t*> apicxt_sequence_map;
    std::multimap<pos_u64id_t, POSHandle*> missing_handle_map;

    if(unlikely(POS_SUCCESS != (
        retval = this->init_handle_managers(is_restoring)
    ))){
        POS_WARN_C("failed to initialize handle managers");
        goto exit;
    }
    
    if(unlikely(POS_SUCCESS != (
        retval = this->__create_qgroup()
    ))){
        POS_WARN_C("failed to initialize queue group");
        goto exit;
    }

exit:
    if(unlikely(retval != POS_SUCCESS)){
        this->status = kPOS_ClientStatus_Hang;
    } else {
        /*!
         *  \brief  enable parser and worker to poll
         *  \note   we won't enable client to be active while restoring,
         *          this should be done after all unexecuted API are loaded
         *          again to parser2worker apicxt queues.
         */
        if(is_restoring == true){
            this->status = kPOS_ClientStatus_Hang;
        } else {
            this->status = kPOS_ClientStatus_Active;
        }
    }
}


void POSClient::deinit(){
    // deinit handle manager of the client
    this->deinit_handle_managers();

    // if client is under trace mode, we dump all its handles
    if(this->_cxt.trace_resource){
        if(unlikely(POS_SUCCESS != this->persist_handles(/* with_state */false))){
            POS_WARN_C("failed to persist handle for tracing");
        }
    }

    // stop parser and worker to poll
    this->status = kPOS_ClientStatus_Hang;

    // destory queue group
    this->__destory_qgroup();

    // shutdown parser and worker
    if(this->parser != nullptr){ delete this->parser; }
    if(this->worker != nullptr){ delete this->worker; }

exit:
    ;
}


pos_retval_t POSClient::persist(std::string& ckpt_dir){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSClient client_binary;
    std::ofstream ckpt_file_stream;
    std::string ckpt_file_path;

    POS_ASSERT(ckpt_dir.size() > 0);

    // verify the path exists
    if(unlikely(!std::filesystem::exists(ckpt_dir))){
        POS_WARN_C(
            "failed to persist client state, no ckpt directory exists, this is a bug: ckpt_dir(%s)",
            ckpt_dir.c_str()
        );
        retval = POS_FAILED_NOT_EXIST;
        goto exit;
    }

    // record client state
    client_binary.set_uuid(this->id);
    client_binary.set_pid(this->pid);
    client_binary.set_job_name(this->_cxt.job_name);
    client_binary.set_api_inst_pc(this->_api_inst_pc);

    // form the path to the checkpoint file of this handle
    ckpt_file_path = ckpt_dir + std::string("/c.bin");

    // write to file
    ckpt_file_stream.open(ckpt_file_path, std::ios::binary | std::ios::out);
    if(!ckpt_file_stream){
        POS_WARN_C(
            "failed to dump client to file, failed to open file: path(%s)",
            ckpt_file_path.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }
    if(!client_binary.SerializeToOstream(&ckpt_file_stream)){
        POS_WARN_C(
            "failed to dump client to file, protobuf failed to dump: path(%s)",
            ckpt_file_path.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }

exit:
    if(ckpt_file_stream.is_open()){ ckpt_file_stream.close(); }
    return retval;
}


pos_retval_t POSClient::restore_handles(std::string& ckpt_dir){
    pos_retval_t retval = POS_SUCCESS, dirty_retval = POS_SUCCESS;
    uint64_t i;
    std::tuple<pos_resource_typeid_t, pos_u64id_t> handle_info;
    std::map<pos_resource_typeid_t, std::vector<pos_u64id_t>> handle_map;

    std::vector<POSHandle*> handle_list;
    typename std::map<pos_resource_typeid_t, std::vector<pos_u64id_t>>::iterator map_iter;
    POSHandle *handle;

    #if POS_CONF_EVAL_CkptOptLevel == 1
        std::map<pos_resource_typeid_t, double> restore_handle_ms;
        std::map<pos_resource_typeid_t, double> restore_handle_state_ms;
        typename std::map<pos_resource_typeid_t, double>::iterator timer_map_iter;
        uint64_t s_tick, e_tick;
    #endif

    auto __deassemble_file_name = [](const std::string& filename) -> std::tuple<pos_resource_typeid_t, pos_u64id_t> {
        std::string baseName = filename.substr(0, filename.find_last_of('.'));
        std::stringstream ss(baseName);
        std::string part;
        std::vector<std::string> parts;

        while (std::getline(ss, part, '-')) { parts.push_back(part); }
        POS_ASSERT(parts.size() == 3);
        POS_ASSERT(parts[0] == std::string("h"));
        
        return std::make_tuple(
            std::stoul(parts[1]),
            std::stoull(parts[2])
        );
    };

    POS_ASSERT(ckpt_dir.size() > 0);
    if (!std::filesystem::exists(ckpt_dir) || !std::filesystem::is_directory(ckpt_dir)) {
        POS_WARN_C("failed to restore handles, ckpt directory not exist: %s", ckpt_dir.c_str())
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    // reallocate handles in the handle manager
    for (const auto& entry : std::filesystem::directory_iterator(ckpt_dir)) {
        if (    entry.is_regular_file() 
            &&  entry.path().extension() == ".bin"
            &&  entry.path().filename().string().rfind("h-", 0) == 0
        ){
            handle_info = __deassemble_file_name(entry.path().filename().string());
            retval = this->__reallocate_single_handle(
                /* ckpt_file */ entry.path().string(),
                /* rid */ std::get<0>(handle_info),
                /* hid */ std::get<1>(handle_info)
            );
            if(unlikely(retval != POS_SUCCESS)){
                dirty_retval = retval;
                POS_WARN_C(
                    "failed to restore handle: rid(%u), hid(%lu), retval(%u)",
                    std::get<0>(handle_info),
                    std::get<1>(handle_info),
                    retval
                );
                continue;
            }
            handle_map[std::get<0>(handle_info)].push_back(std::get<1>(handle_info));
            POS_DEBUG_C("restored handle: rid(%lu), hid(%lu)", std::get<0>(handle_info), std::get<1>(handle_info));

            #if POS_CONF_EVAL_CkptOptLevel == 1
                if(unlikely(restore_handle_ms.count(std::get<0>(handle_info)) == 0)){
                    restore_handle_ms[std::get<0>(handle_info)] = 0.0;
                }
                if(unlikely(restore_handle_state_ms.count(std::get<0>(handle_info)) == 0)){
                    restore_handle_state_ms[std::get<0>(handle_info)] = 0.0;
                }
            #endif
        }
    }

    // reassign each handle's parent handles
    for(map_iter = handle_map.begin(); map_iter != handle_map.end(); map_iter++){
        POS_CHECK_POINTER(this->handle_managers[map_iter->first]);
        for(i=0; i<map_iter->second.size(); i++){
            handle = this->handle_managers[map_iter->first]->get_handle_by_id(map_iter->second[i]);
            if(unlikely(handle == nullptr)){
                continue;
            }
            retval = this->__reassign_handle_parents(handle);
            if(unlikely(retval != POS_SUCCESS)){
                dirty_retval = retval;
                POS_WARN_C("failed to reassign handle parents: rid(%u), hid(%lu)", map_iter->first, map_iter->second);
                goto exit;
            }
            handle_list.push_back(handle);
        }
    }

    /*!
     *  \note   [1] under baseline C/R, we directly resume both the resource and state here
     *          [2] under PhOS C/R, we will on-demand resume resource and its state
     */
    #if POS_CONF_EVAL_CkptOptLevel == 1
        for(i=0; i<handle_list.size(); i++){
            POS_CHECK_POINTER(handle = handle_list[i]);

            // restore handle
            s_tick = POSUtilTscTimer::get_tsc();
            retval = handle->restore();
            if(unlikely(retval != POS_SUCCESS)){
                dirty_retval = retval;
                POS_WARN_C(
                    "failed to restore resource on device: client_addr(%p), rid(%u)",
                    handle->client_addr,
                    handle->resource_type_id
                );
                goto exit;
            }
            e_tick = POSUtilTscTimer::get_tsc();
            restore_handle_ms[handle->resource_type_id] += this->_ws->tsc_timer.tick_to_ms(e_tick-s_tick);

            // restore state
            s_tick = POSUtilTscTimer::get_tsc();
            if(handle->state_size > 0){
                retval = handle->reload_state( /* stream_id */ 0);
                if(unlikely(retval != POS_SUCCESS)){
                    dirty_retval = retval;
                    POS_WARN_C(
                        "failed to restore resource state on device: client_addr(%p), rid(%u)",
                        handle->client_addr,
                        handle->resource_type_id
                    );
                    goto exit;
                }
            }
            e_tick = POSUtilTscTimer::get_tsc();
            restore_handle_state_ms[handle->resource_type_id] += this->_ws->tsc_timer.tick_to_ms(e_tick-s_tick);
        }

        // print restore duration information
        for(timer_map_iter = restore_handle_ms.begin(); timer_map_iter != restore_handle_ms.end(); timer_map_iter++){
            POS_LOG_C("restore handle: rid(%lu), duration(%lf ms)", timer_map_iter->first, timer_map_iter->second);
        }
        for(timer_map_iter = restore_handle_state_ms.begin(); timer_map_iter != restore_handle_state_ms.end(); timer_map_iter++){
            POS_LOG_C("restore handle state: rid(%lu), duration(%lf ms)", timer_map_iter->first, timer_map_iter->second);
        }
    #elif POS_CONF_EVAL_CkptOptLevel == 2
        /* nothing */
    #endif

exit:
    return dirty_retval;
}


pos_retval_t POSClient::restore_apicxts(std::string& ckpt_dir){
    pos_retval_t retval = POS_SUCCESS;
    pos_u64id_t apicxt_id;
    std::set<std::filesystem::path> sorted_unexecuted_apicxts;
    # if POS_CONF_EVAL_CkptOptLevel == 2
        std::set<std::filesystem::path> sorted_recomputation_apicxts;
    #endif
    typename std::set<std::filesystem::path>::iterator set_iter;

    POS_ASSERT(ckpt_dir.size() > 0);
    if (!std::filesystem::exists(ckpt_dir) || !std::filesystem::is_directory(ckpt_dir)) {
        POS_WARN_C("failed to restore api contexts, ckpt directory not exist: %s", ckpt_dir.c_str())
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    # if POS_CONF_EVAL_CkptOptLevel == 2
        // enqueue recomputation apis
        for (const auto& entry : std::filesystem::directory_iterator(ckpt_dir)) {
            if (    entry.is_regular_file() 
                &&  entry.path().extension() == ".bin"
                &&  entry.path().filename().string().rfind("ra-", 0) == 0
            ){
                sorted_recomputation_apicxts.insert(entry.path());
            }
        }
        for(set_iter = sorted_recomputation_apicxts.begin(); set_iter != sorted_recomputation_apicxts.end(); set_iter++){
            retval = this->__reload_apicxt((*set_iter).string(), ApiCxt_TypeId_Recomputation);
            if(unlikely(retval != POS_SUCCESS)){
                POS_WARN_C(
                    "failed to reload recomputation api context: ckpt_file(%s)",
                    (*set_iter).string().c_str()
                );
                goto exit;
            }
        }
    #endif

    // enqueue unexecuted apis
    for (const auto& entry : std::filesystem::directory_iterator(ckpt_dir)) {
        if (    entry.is_regular_file() 
            &&  entry.path().extension() == ".bin"
            &&  entry.path().filename().string().rfind("ua-", 0) == 0
        ){
            sorted_unexecuted_apicxts.insert(entry.path());
        }
    }
    for(set_iter = sorted_unexecuted_apicxts.begin(); set_iter != sorted_unexecuted_apicxts.end(); set_iter++){
        retval = this->__reload_apicxt((*set_iter).string(), ApiCxt_TypeId_Unexecuted);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN_C(
                "failed to reload unexecuted api context: ckpt_file(%s)",
                (*set_iter).string().c_str()
            );
            goto exit;
        }
    }

exit:
    return retval;
}


template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
pos_retval_t POSClient::push_q(void *qe){
    pos_retval_t retval = POS_SUCCESS;
    POSAPIContext_QE_t *apictx_qe;
    POSCommand_QE_t *cmd_qe;

    static_assert(
            qtype == kPOS_QueueType_ApiCxt_WQ || qtype == kPOS_QueueType_ApiCxt_CQ
        ||  qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ
        ||  qtype == kPOS_QueueType_ApiCxt_Trace_WQ
        ||  qtype == kPOS_QueueType_Cmd_WQ || qtype == kPOS_QueueType_Cmd_CQ,
        "unknown queue type obtained"
    );

    POS_CHECK_POINTER(qe);

    // api context worker queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_WQ){
        POS_CHECK_POINTER(apictx_qe = reinterpret_cast<POSAPIContext_QE_t*>(qe));
        
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Parser2Worker,
            "ApiCxt_WQE can only be pushed to rpc2parser or parser2worker queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            this->_apicxt_rpc2parser_wq->push(apictx_qe);
        } else { // qdir == kPOS_QueueDirection_Parser2Worker
            this->_apicxt_parser2worker_wq->push(apictx_qe);
        }
    }

    // api context completion queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CQ){
        POS_CHECK_POINTER(apictx_qe = reinterpret_cast<POSAPIContext_QE_t*>(qe));
 
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Rpc2Worker,
            "ApiCxt_CQE can only be pushed to rpc2parser or rpc2worker queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            this->_apicxt_rpc2parser_cq->push(apictx_qe);
        } else { // qdir == kPOS_QueueDirection_Rpc2Worker
            this->_apicxt_rpc2worker_cq->push(apictx_qe);
        }
    }

    // api context ckptdag queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ) {
        POS_CHECK_POINTER(apictx_qe = reinterpret_cast<POSAPIContext_QE_t*>(qe));

        static_assert(
            qdir == kPOS_QueueDirection_WorkerLocal,
            "ApiCxt_CkptDag_WQE can only be pushed to worker local queue"
        );

        this->_apicxt_workerlocal_ckptdag_wq->push(apictx_qe);
    }

    // api context trace queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_Trace_WQ) {
        POS_CHECK_POINTER(apictx_qe = reinterpret_cast<POSAPIContext_QE_t*>(qe));

        static_assert(
            qdir == kPOS_QueueDirection_ParserLocal,
            "ApiCxt_Trace_WQE can only be pushed to parser local queue"
        );

        this->_apicxt_parserlocal_trace_wq->push(apictx_qe);
    }

    // command work queue
    if constexpr (qtype == kPOS_QueueType_Cmd_WQ){
        POS_CHECK_POINTER(cmd_qe = reinterpret_cast<POSCommand_QE_t*>(qe));
        
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "Cmd_WQE can only be pushed to parser2worker or oob2parser queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            this->_cmd_parser2worker_wq->push(cmd_qe);
        } else { // qdir == kPOS_QueueDirection_Oob2Parser
            this->_cmd_oob2parser_wq->push(cmd_qe);
        }
    }

    // command completion queue
    if constexpr (qtype == kPOS_QueueType_Cmd_CQ){
        POS_CHECK_POINTER(cmd_qe = reinterpret_cast<POSCommand_QE_t*>(qe));
        
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "Cmd_CQE can only be pushed to parser2worker or oob2parser queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            this->_cmd_parser2worker_cq->push(cmd_qe);
        } else { // qdir == kPOS_QueueDirection_Oob2Parser
            this->_cmd_oob2parser_cq->push(cmd_qe);
        }
    }

exit:
    return retval;
}
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_CQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_ApiCxt_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Rpc2Worker, kPOS_QueueType_ApiCxt_CQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_WorkerLocal, kPOS_QueueType_ApiCxt_CkptDag_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_ParserLocal, kPOS_QueueType_ApiCxt_Trace_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_CQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_WQ>(void *qe);
template pos_retval_t POSClient::push_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_CQ>(void *qe);


template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
pos_retval_t POSClient::clear_q(){
    pos_retval_t retval = POS_SUCCESS;

    static_assert(
            qtype == kPOS_QueueType_ApiCxt_WQ || qtype == kPOS_QueueType_ApiCxt_CQ
        ||  qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ
        ||  qtype == kPOS_QueueType_ApiCxt_Trace_WQ
        ||  qtype == kPOS_QueueType_Cmd_WQ || qtype == kPOS_QueueType_Cmd_CQ,
        "unknown queue type obtained"
    );

    // api context worker queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Parser2Worker,
            "ApiCxt_WQE can only be located within rpc2parser or parser2worker queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            this->_apicxt_rpc2parser_wq->drain();
        } else { // qdir == kPOS_QueueDirection_Parser2Worker
            this->_apicxt_parser2worker_wq->drain();
        }
    }

    // api context completion queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CQ){
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Rpc2Worker,
            "ApiCxt_CQE can only be located within rpc2parser or rpc2worker queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            this->_apicxt_rpc2parser_cq->drain();
        } else { // qdir == kPOS_QueueDirection_Rpc2Worker
            this->_apicxt_rpc2worker_cq->drain();
        }
    }

    // api context ckptdag queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ) {
        static_assert(
            qdir == kPOS_QueueDirection_WorkerLocal,
            "ApiCxt_CkptDag_WQE can only be located within worker local queue"
        );
        this->_apicxt_workerlocal_ckptdag_wq->drain();
    }

    // api context trace queue 
    if constexpr (qtype == kPOS_QueueType_ApiCxt_Trace_WQ) {
        static_assert(
            qdir == kPOS_QueueDirection_ParserLocal,
            "ApiCxt_CkptDag_WQE can only be located within parser local queue"
        );
        this->_apicxt_parserlocal_trace_wq->drain();
    }

    // command work queue
    if constexpr (qtype == kPOS_QueueType_Cmd_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "Cmd_WQE can only be located within parser2worker or oob2parser queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            this->_cmd_parser2worker_wq->drain();
        } else { // qdir == kPOS_QueueDirection_Oob2Parser
            this->_cmd_oob2parser_wq->drain();
        }
    }

    // command completion queue
    if constexpr (qtype == kPOS_QueueType_Cmd_CQ){
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "Cmd_CQE can only be located within parser2worker or oob2parser queue"
        );

        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            this->_cmd_parser2worker_cq->drain();
        } else { // qdir == kPOS_QueueDirection_Oob2Parser
            this->_cmd_oob2parser_cq->drain();
        }
    }

exit:
    return retval;
}
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_CQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_ApiCxt_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Rpc2Worker, kPOS_QueueType_ApiCxt_CQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_WorkerLocal, kPOS_QueueType_ApiCxt_CkptDag_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_ParserLocal, kPOS_QueueType_ApiCxt_Trace_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_CQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_WQ>();
template pos_retval_t POSClient::clear_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_CQ>();


template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
pos_retval_t POSClient::poll_q(std::vector<POSAPIContext_QE*>* qes){
    pos_retval_t retval = POS_SUCCESS;
    POSAPIContext_QE *apicxt_qe;
    POSLockFreeQueue<POSAPIContext_QE_t*> *apicxt_q;

    static_assert(
            qtype == kPOS_QueueType_ApiCxt_WQ 
        ||  qtype == kPOS_QueueType_ApiCxt_CQ 
        ||  qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ
        ||  qtype == kPOS_QueueType_ApiCxt_Trace_WQ,
        "invalid queue type obtained"
    );

    POS_CHECK_POINTER(qes);

    // api context work queue
    if constexpr (qtype == kPOS_QueueType_ApiCxt_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Parser2Worker,
            "POSAPIContext_WQE can only be poll from rpc2parser or parser2worker queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            apicxt_q = this->_apicxt_rpc2parser_wq;
        } else { // kPOS_QueueDirection_Parser2Worker
            apicxt_q = this->_apicxt_parser2worker_wq;
        }
    }

    // api context completion queue
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CQ){
        static_assert(
            qdir == kPOS_QueueDirection_Rpc2Parser || qdir == kPOS_QueueDirection_Rpc2Worker,
            "POSAPIContext_CQE can only be poll from rpc2parser or parser2worker queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Rpc2Parser){
            apicxt_q = this->_apicxt_rpc2parser_cq;
        } else { // kPOS_QueueDirection_Rpc2Worker
            apicxt_q = this->_apicxt_rpc2worker_cq;
        }
    }

    // api context ckptdag work queue
    if constexpr (qtype == kPOS_QueueType_ApiCxt_CkptDag_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_WorkerLocal,
            "ApiCxt_CkptDag_WQE can only be passed within worker local queue"
        );
        apicxt_q = this->_apicxt_workerlocal_ckptdag_wq;
    }

    // api context trace work queue
    if constexpr (qtype == kPOS_QueueType_ApiCxt_Trace_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_ParserLocal,
            "ApiCxt_CkptDag_WQE can only be passed within parser local queue"
        );
        apicxt_q = this->_apicxt_parserlocal_trace_wq;
    }

    POS_CHECK_POINTER(apicxt_q);
    while(POS_SUCCESS == apicxt_q->dequeue(apicxt_qe)){
        qes->push_back(apicxt_qe);
    }

exit:
    return retval;
}
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_WQ>(std::vector<POSAPIContext_QE*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_ApiCxt_WQ>(std::vector<POSAPIContext_QE*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_WorkerLocal, kPOS_QueueType_ApiCxt_CkptDag_WQ>(std::vector<POSAPIContext_QE*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_ParserLocal, kPOS_QueueType_ApiCxt_Trace_WQ>(std::vector<POSAPIContext_QE*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Rpc2Parser, kPOS_QueueType_ApiCxt_CQ>(std::vector<POSAPIContext_QE*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Rpc2Worker, kPOS_QueueType_ApiCxt_CQ>(std::vector<POSAPIContext_QE*>* qes);


template<pos_queue_direction_t qdir, pos_queue_type_t qtype>
pos_retval_t POSClient::poll_q(std::vector<POSCommand_QE_t*>* qes){
    pos_retval_t retval = POS_SUCCESS;
    POSCommand_QE_t *cmd_qe;
    POSLockFreeQueue<POSCommand_QE_t*> *cmd_q;
    
    static_assert(
        qtype == kPOS_QueueType_Cmd_WQ || qtype == kPOS_QueueType_Cmd_CQ,
        "invalid queue type obtained"
    );

    POS_CHECK_POINTER(qes);

    // command work queue
    if constexpr (qtype == kPOS_QueueType_Cmd_WQ){
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "POSCommand_WQE can only be polled from parser2worker or oob2parser queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            cmd_q = this->_cmd_parser2worker_wq;
        } else { // kPOS_QueueDirection_Oob2Parser
            cmd_q = this->_cmd_oob2parser_wq;
        }
    }

    // command completion queue
    if constexpr (qtype == kPOS_QueueType_Cmd_CQ){
        static_assert(
            qdir == kPOS_QueueDirection_Parser2Worker || qdir == kPOS_QueueDirection_Oob2Parser,
            "POSCommand_CQE can only be polled from parser2worker or oob2parser queue"
        );
        if constexpr (qdir == kPOS_QueueDirection_Parser2Worker){
            cmd_q = this->_cmd_parser2worker_cq;
        } else { // kPOS_QueueDirection_Oob2Parser
            cmd_q = this->_cmd_oob2parser_cq;
        }
    }

    POS_CHECK_POINTER(cmd_q);
    while(POS_SUCCESS == cmd_q->dequeue(cmd_qe)){
        qes->push_back(cmd_qe);
    }

exit:
    return retval;
}
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_WQ>(std::vector<POSCommand_QE_t*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_WQ>(std::vector<POSCommand_QE_t*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_Cmd_CQ>(std::vector<POSCommand_QE_t*>* qes);
template pos_retval_t POSClient::poll_q<kPOS_QueueDirection_Oob2Parser, kPOS_QueueType_Cmd_CQ>(std::vector<POSCommand_QE_t*>* qes);


pos_retval_t POSClient::__create_qgroup(){
    pos_retval_t retval = POS_SUCCESS;

    // rpc2parser apicxt work queue
    this->_apicxt_rpc2parser_wq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_rpc2parser_wq);
    POS_DEBUG_C("created rpc2parser apicxt WQ: uuid(%lu)", this->id);

    // rpc2parser apicxt completion queue
    this->_apicxt_rpc2parser_cq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_rpc2parser_cq);
    POS_DEBUG_C("created rpc2parser apicxt CQ: uuid(%lu)", this->id);

    // parser2worker apicxt work queue
    this->_apicxt_parser2worker_wq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_parser2worker_wq);
    POS_DEBUG_C("created parser2worker apicxt WQ: uuid(%lu)", this->id);

    // rpc2worker apicxt completion queue
    this->_apicxt_rpc2worker_cq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_rpc2worker_cq);
    POS_DEBUG_C("created rpc2worker apicxt CQ: uuid(%lu)", this->id);

    // workerlocal apicxt ckptdag queue
    this->_apicxt_workerlocal_ckptdag_wq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_workerlocal_ckptdag_wq);
    POS_DEBUG_C("created workerlocal ckptdag apicxt WQ: uuid(%lu)", this->id);

    // parserlocal apicxt trace queue
    this->_apicxt_parserlocal_trace_wq = new POSLockFreeQueue<POSAPIContext_QE_t*>();
    POS_CHECK_POINTER(this->_apicxt_parserlocal_trace_wq);
    POS_DEBUG_C("created parserlocal trace apicxt WQ: uuid(%lu)", this->id);

    // parser2worker cmd work queue
    this->_cmd_parser2worker_wq = new POSLockFreeQueue<POSCommand_QE_t*>();
    POS_CHECK_POINTER(this->_cmd_parser2worker_wq);
    POS_DEBUG_C("created parser2worker cmd WQ: uuid(%lu)", this->id);

    // parser2worker cmd completion queue
    this->_cmd_parser2worker_cq = new POSLockFreeQueue<POSCommand_QE_t*>();
    POS_CHECK_POINTER(this->_cmd_parser2worker_cq);
    POS_DEBUG_C("created parser2worker cmd CQ: uuid(%lu)", this->id);

    // oob2parser cmd work queue
    this->_cmd_oob2parser_wq = new POSLockFreeQueue<POSCommand_QE_t*>();
    POS_CHECK_POINTER(this->_cmd_oob2parser_wq);
    POS_DEBUG_C("created oob2parser cmd WQ: uuid(%lu)", this->id);

    // oob2parser cmd completion queue
    this->_cmd_oob2parser_cq = new POSLockFreeQueue<POSCommand_QE_t*>();
    POS_CHECK_POINTER(this->_cmd_oob2parser_cq);
    POS_DEBUG_C("created oob2parser cmd CQ: uuid(%lu)", this->id);

    return retval;
}


pos_retval_t POSClient::__destory_qgroup(){
    pos_retval_t retval = POS_SUCCESS;
    
    // rpc2parser apicxt work queue
    POS_CHECK_POINTER(this->_apicxt_rpc2parser_wq);
    this->_apicxt_rpc2parser_wq->lock();
    delete this->_apicxt_rpc2parser_wq;
    POS_DEBUG_C("destoryed rpc2parser apicxt WQ: uuid(%lu)", this->id);

    // rpc2parser apicxt completion queue
    POS_CHECK_POINTER(this->_apicxt_rpc2parser_cq);
    this->_apicxt_rpc2parser_cq->lock();
    delete this->_apicxt_rpc2parser_cq;
    POS_DEBUG_C("destoryed rpc2parser apicxt CQ: uuid(%lu)", this->id);

    // parser2worker apicxt work queue
    POS_CHECK_POINTER(this->_apicxt_parser2worker_wq);
    this->_apicxt_parser2worker_wq->lock();
    delete this->_apicxt_parser2worker_wq;
    POS_DEBUG_C("destoryed parser2worker apicxt WQ: uuid(%lu)", this->id);

    // rpc2worker apicxt completion queue
    POS_CHECK_POINTER(this->_apicxt_rpc2worker_cq);
    this->_apicxt_rpc2worker_cq->lock();
    delete this->_apicxt_rpc2worker_cq;
    POS_DEBUG_C("destoryed rpc2worker apicxt CQ: uuid(%lu)", this->id);

    // workerlocal ckptdag apicxt work queue
    POS_CHECK_POINTER(this->_apicxt_workerlocal_ckptdag_wq);
    this->_apicxt_workerlocal_ckptdag_wq->lock();
    delete this->_apicxt_workerlocal_ckptdag_wq;
    POS_DEBUG_C("destoryed workerlocal_ckptdag apicxt WQ: uuid(%lu)", this->id);

    // parserlocal trace apicxt work queue
    POS_CHECK_POINTER(this->_apicxt_parserlocal_trace_wq);
    this->_apicxt_parserlocal_trace_wq->lock();
    delete this->_apicxt_parserlocal_trace_wq;
    POS_DEBUG_C("destoryed parserlocal trace apicxt WQ: uuid(%lu)", this->id);

    // parser2worker cmd work queue
    POS_CHECK_POINTER(this->_cmd_parser2worker_wq);
    this->_cmd_parser2worker_wq->lock();
    delete this->_cmd_parser2worker_wq;
    POS_DEBUG_C("destoryed parser2worker apicxt WQ: uuid(%lu)", this->id);

    // parser2worker cmd completion queue
    POS_CHECK_POINTER(this->_cmd_parser2worker_cq);
    this->_cmd_parser2worker_cq->lock();
    delete this->_cmd_parser2worker_cq;
    POS_DEBUG_C("destoryed parser2worker cmd CQ: uuid(%lu)", this->id);

    // oob2parser cmd work queue
    POS_CHECK_POINTER(this->_cmd_oob2parser_wq);
    this->_cmd_oob2parser_wq->lock();
    delete this->_cmd_oob2parser_wq;
    POS_DEBUG_C("destoryed oob2parser cmd WQ: uuid(%lu)", this->id);

    // oob2parser cmd completion queue
    POS_CHECK_POINTER(this->_cmd_oob2parser_cq);
    this->_cmd_oob2parser_cq->lock();
    delete this->_cmd_oob2parser_cq;
    POS_DEBUG_C("destoryed oob2parser cmd CQ: uuid(%lu)", this->id);

exit:
    return retval;
}


pos_retval_t POSClient::__reload_apicxt(const std::string& ckpt_file, pos_apicxt_typeid_t type){
    pos_retval_t retval = POS_SUCCESS;
    POSAPIContext_QE_t *apicxt;
    uint64_t i;
    pos_resource_typeid_t rid;
    pos_u64id_t hid;
    POSHandle *handle;

    POS_ASSERT(ckpt_file.size() > 0);
    
    POS_CHECK_POINTER(apicxt = new POSAPIContext_QE_t(this, ckpt_file, type));
    if(unlikely(apicxt->client == nullptr)){
        POS_WARN_C(
            "failed to restore apicxt from binary checkpoint file: ckpt_file(%s)",
            ckpt_file.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }

    // restore handle pointer inside handle views
    for(i=0; i<apicxt->input_handle_views.size(); i++){
        rid = apicxt->input_handle_views[i].resource_type_id;
        hid = apicxt->input_handle_views[i].id;
        POS_CHECK_POINTER(this->handle_managers[rid]);
        POS_CHECK_POINTER(handle = this->handle_managers[rid]->get_handle_by_id(hid));
        apicxt->input_handle_views[i].handle = handle;
    }
    for(i=0; i<apicxt->output_handle_views.size(); i++){
        rid = apicxt->output_handle_views[i].resource_type_id;
        hid = apicxt->output_handle_views[i].id;
        POS_CHECK_POINTER(this->handle_managers[rid]);
        POS_CHECK_POINTER(handle = this->handle_managers[rid]->get_handle_by_id(hid));
        apicxt->output_handle_views[i].handle = handle;
    }
    for(i=0; i<apicxt->inout_handle_views.size(); i++){
        rid = apicxt->inout_handle_views[i].resource_type_id;
        hid = apicxt->inout_handle_views[i].id;
        POS_CHECK_POINTER(this->handle_managers[rid]);
        POS_CHECK_POINTER(handle = this->handle_managers[rid]->get_handle_by_id(hid));
        apicxt->inout_handle_views[i].handle = handle;
    }
    for(i=0; i<apicxt->create_handle_views.size(); i++){
        rid = apicxt->create_handle_views[i].resource_type_id;
        hid = apicxt->create_handle_views[i].id;
        POS_CHECK_POINTER(this->handle_managers[rid]);
        POS_CHECK_POINTER(handle = this->handle_managers[rid]->get_handle_by_id(hid));
        apicxt->create_handle_views[i].handle = handle;
    }
    for(i=0; i<apicxt->delete_handle_views.size(); i++){
        rid = apicxt->delete_handle_views[i].resource_type_id;
        hid = apicxt->delete_handle_views[i].id;
        POS_CHECK_POINTER(this->handle_managers[rid]);
        POS_CHECK_POINTER(handle = this->handle_managers[rid]->get_handle_by_id(hid));
        apicxt->delete_handle_views[i].handle = handle;
    }

    // push this wqe to worker
    this->template push_q<kPOS_QueueDirection_Parser2Worker, kPOS_QueueType_ApiCxt_WQ>(apicxt);

exit:
    return retval;
}
