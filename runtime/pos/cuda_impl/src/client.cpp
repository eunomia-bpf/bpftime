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

#include <iostream>
#include <set>
#include <filesystem>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/client.h"
#include "pos/include/transport.h"
#include "pos/include/handle.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/handle.h"


POSClient_CUDA::POSClient_CUDA(pos_client_uuid_t id, pid_t pid, pos_client_cxt_CUDA_t cxt, POSWorkspace *ws) 
        : POSClient(id, pid, cxt.cxt_base, ws), _cxt_CUDA(cxt)
{
    // raise parser thread
    this->parser = new POSParser_CUDA(ws, this);
    POS_CHECK_POINTER(this->parser);
    this->parser->init();

    // raise worker thread
    this->worker = new POSWorker_CUDA(ws, this);
    POS_CHECK_POINTER(this->worker);
    this->worker->init();

    if(unlikely(POS_SUCCESS != this->init_transport())){
        POS_WARN_C("failed to initialize transport for client %lu, migration would be failed", id);
    }
}


POSClient_CUDA::POSClient_CUDA(){}


POSClient_CUDA::~POSClient_CUDA(){}


pos_retval_t POSClient_CUDA::init_handle_managers(bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    std::vector<POSHandle_CUDA_Device*> device_handles;
    std::vector<POSHandle_CUDA_Context*> context_handles;

    POSHandleManager_CUDA_Device *device_mgr;
    POSHandleManager_CUDA_Context *ctx_mgr;
    POSHandleManager_CUDA_Stream *stream_mgr;
    POSHandleManager_cuBLAS_Context *cublas_context_mgr;
    POSHandleManager_CUDA_Event *event_mgr;
    POSHandleManager_CUDA_Module *module_mgr;
    POSHandleManager_CUDA_Function *function_mgr;
    POSHandleManager_CUDA_Var *var_mgr;
    POSHandleManager_CUDA_Memory *memory_mgr;

    std::map<uint64_t, std::vector<POSHandle*>> related_handles;

    auto __cast_to_base_handle_list = [](auto handle_list) -> std::vector<POSHandle*> {
        std::vector<POSHandle*> ret_list;
        for(auto handle : handle_list){ ret_list.push_back(handle); }
        return ret_list;
    };


    /*!
     *  \note   Hierarchy of CUDA Resources
         ╔══════════════════════════════════════════════════════════════════════╗
        ╔══════════════════════════════════════════════════════════════════════╗║
        ║                              CUDA Device                             ║║
        ╠══════════════════════════════════════════════════════════════════════╣║
        ║                             CUDA Context                             ║║
        ╠════════════════╦════════════╦══════════════════════════╦═════════════╣║
        ║   CUDA Stream  ║            ║        CUDA Module       ║             ║║
        ╠════════════════╣ CUDA Event ╠═══════════════╦══════════╣ CUDA Memory ║║
        ║ cuBLAS Context ║            ║ CUDA Function ║ CUDA Var ║             ║╝
        ╚════════════════╩════════════╩═══════════════╩══════════╩═════════════╝
     */


    // CUDA device handle manager
    related_handles.clear();
    POS_CHECK_POINTER(device_mgr = new POSHandleManager_CUDA_Device());
    if(unlikely(POS_SUCCESS != (
        retval = device_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA device handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Device] = (POSHandleManager<POSHandle>*)(device_mgr);

    // CUDA context handle manager
    related_handles.clear();
    device_handles = device_mgr->get_handles();
    related_handles.insert({ kPOS_ResourceTypeId_CUDA_Device, __cast_to_base_handle_list(device_handles) });
    POS_CHECK_POINTER(ctx_mgr = new POSHandleManager_CUDA_Context());
    if(unlikely(POS_SUCCESS != (
        retval = ctx_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA context handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Context] = (POSHandleManager<POSHandle>*)(ctx_mgr);

    // CUDA stream handle manager
    related_handles.clear();
    context_handles = ctx_mgr->get_handles();
    related_handles.insert({ kPOS_ResourceTypeId_CUDA_Context, __cast_to_base_handle_list(context_handles) });
    POS_CHECK_POINTER(stream_mgr = new POSHandleManager_CUDA_Stream());
    if(unlikely(POS_SUCCESS != (
        retval = stream_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA stream handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Stream] = (POSHandleManager<POSHandle>*)(stream_mgr);

    // cuBLAS context handle manager
    related_handles.clear();
    POS_CHECK_POINTER(cublas_context_mgr = new POSHandleManager_cuBLAS_Context());
    if(unlikely(POS_SUCCESS != (
        retval = cublas_context_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize cuBLAS context handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_cuBLAS_Context] = (POSHandleManager<POSHandle>*)(cublas_context_mgr);

    // CUDA event handle manager
    related_handles.clear();
    POS_CHECK_POINTER(event_mgr = new POSHandleManager_CUDA_Event());
    if(unlikely(POS_SUCCESS != (
        retval = event_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA event handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Event] = (POSHandleManager<POSHandle>*)(event_mgr);

    // CUDA module handle manager
    related_handles.clear();
    POS_CHECK_POINTER(module_mgr = new POSHandleManager_CUDA_Module());
    if(unlikely(POS_SUCCESS != (
        retval = module_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA module handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Module] = (POSHandleManager<POSHandle>*)(module_mgr);
    if(std::filesystem::exists(this->_cxt.kernel_meta_path) && !is_restoring){
        POS_DEBUG_C("loading kernel meta from cache %s...", this->_cxt.kernel_meta_path.c_str());
        retval = module_mgr->load_cached_function_metas(this->_cxt.kernel_meta_path);
        if(likely(retval == POS_SUCCESS)){
            this->_cxt.is_load_kernel_from_cache = true;
            POS_BACK_LINE
            POS_DEBUG_C("loading kernel meta from cache %s [done]", this->_cxt.kernel_meta_path.c_str());
        } else {
            POS_WARN_C("loading kernel meta from cache %s [failed]", this->_cxt.kernel_meta_path.c_str());
        }
    }

    // CUDA function handle manager
    related_handles.clear();
    POS_CHECK_POINTER(function_mgr = new POSHandleManager_CUDA_Function());
    if(unlikely(POS_SUCCESS != (
        retval = function_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA function handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Function] = (POSHandleManager<POSHandle>*)(function_mgr);

    // CUDA var handle manager
    related_handles.clear();
    POS_CHECK_POINTER(var_mgr = new POSHandleManager_CUDA_Var());
    if(unlikely(POS_SUCCESS != (
        retval = var_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA var handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Var] = (POSHandleManager<POSHandle>*)(var_mgr);

    // CUDA memory handle manager
    related_handles.clear();
    context_handles = ctx_mgr->get_handles();
    related_handles.insert({ kPOS_ResourceTypeId_CUDA_Context, __cast_to_base_handle_list(context_handles) });
    POS_CHECK_POINTER(memory_mgr = new POSHandleManager_CUDA_Memory());
    if(unlikely(POS_SUCCESS != (
        retval = memory_mgr->init(related_handles, is_restoring)
    ))){
        POS_WARN_C("failed to initialize CUDA memory handle manager, client won't be run");
        goto exit;
    }
    this->handle_managers[kPOS_ResourceTypeId_CUDA_Memory] = (POSHandleManager<POSHandle>*)(memory_mgr);

exit:
    return retval;
}


/*
    *  \brief  initialization of transport utilities for migration  
    *  \return POS_SUCCESS for successfully initialization
    */
pos_retval_t POSClient_CUDA::init_transport(){
    pos_retval_t retval = POS_SUCCESS;
    
    // TODO: default to use RDMA here, might support other transport later
    this->_transport = new POSTransport_RDMA</* is_server */false>(/* dev_name */ "");
    POS_CHECK_POINTER(this->_transport);

exit:
    return retval;
}


void POSClient_CUDA::deinit_handle_managers(){
    #if POS_CONF_RUNTIME_EnableTrace
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandleManager_CUDA_Module *hm_module;

        POS_CHECK_POINTER(hm_memory = pos_get_client_typed_hm(this, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory));
        POS_CHECK_POINTER(hm_module = pos_get_client_typed_hm(this, kPOS_ResourceTypeId_CUDA_Module, POSHandleManager_CUDA_Module));

        hm_memory->print_metrics();
        hm_module->print_metrics();
    #endif

    this->__dump_hm_cuda_functions();
}


pos_retval_t POSClient_CUDA::persist_handles(bool with_state){
    pos_retval_t retval = POS_SUCCESS;
    std::string trace_dir, apicxt_dir, resource_dir;
    uint64_t i;
    POSHandleManager<POSHandle>* hm;
    POSHandle *handle;
    POSAPIContext_QE *wqe;
    std::vector<POSAPIContext_QE*> wqes;

    POS_LOG_C("dumping trace resource result...");

    // create directory
    retval = this->_ws->ws_conf.get(POSWorkspaceConf::kRuntimeTraceDir, trace_dir);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to obtain directory to store trace result, failed to dump");
        goto exit;
    }
    trace_dir += std::string("/")
                + std::to_string(this->_cxt.pid)
                + std::string("-")
                + std::to_string(this->_ws->tsc_timer.get_tsc());
    apicxt_dir = trace_dir + std::string("/apicxt/");
    resource_dir = trace_dir + std::string("/resource/");
    if (std::filesystem::exists(trace_dir)) { std::filesystem::remove_all(trace_dir); }
    try {
        std::filesystem::create_directories(apicxt_dir);
        std::filesystem::create_directories(resource_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        POS_WARN_C("failed to create directory to store trace result, failed to dump");
        goto exit;
    }
    POS_BACK_LINE;
    POS_LOG_C("dumping trace resource result to %s...", trace_dir.c_str());

    // dumping API context
    wqes.clear();
    this->template poll_q<kPOS_QueueDirection_ParserLocal, kPOS_QueueType_ApiCxt_Trace_WQ>(&wqes);
    for(i=0; i<wqes.size(); i++){
        POS_CHECK_POINTER(wqe = wqes[i]);
        wqe->persist</* with_params */ false, /* type */ ApiCxt_TypeId_Unexecuted>(apicxt_dir);
    }

    // dumping resources
    for(auto &handle_id : this->_ws->resource_type_idx){
        POS_CHECK_POINTER(
            hm = pos_get_client_typed_hm(this, handle_id, POSHandleManager<POSHandle>)
        );
        for(i=0; i<hm->get_nb_handles(); i++){
            POS_CHECK_POINTER(handle = hm->get_handle_by_id(i));
            if(with_state){
                retval = handle->checkpoint_commit_sync(handle->latest_version, /* stream_id */ 0);
                if(unlikely(POS_SUCCESS != retval)){
                    POS_WARN_C("failed to commit the status of handle: rname(%s), hid(%u)", handle->get_resource_name().c_str(), handle->id);
                    retval = POS_FAILED;
                    goto exit;
                }
            }

            retval = handle->checkpoint_persist_sync(resource_dir, with_state, handle->latest_version);
            if(unlikely(POS_SUCCESS != retval)){
                POS_WARN_C("failed to persist handle: rname(%s), hid(%u)", handle->get_resource_name().c_str(), handle->id);
                retval = POS_FAILED;
                goto exit;
            }
        }
    }

    POS_BACK_LINE;
    POS_LOG_C("dumping trace resource result to %s [done]", trace_dir.c_str());

exit:
    return retval;
}


pos_retval_t POSClient_CUDA::__reallocate_single_handle(const std::string& ckpt_file, pos_resource_typeid_t rid, pos_u64id_t hid){
    pos_retval_t retval = POS_SUCCESS;
    POSHandle *restored_handle = nullptr;

    POS_ASSERT(ckpt_file.size() > 0);
    POS_ASSERT(
        std::find(
            this->_ws->resource_type_idx.begin(),
            this->_ws->resource_type_idx.begin(),
            rid
        ) != this->_ws->resource_type_idx.end()
    );
    POS_CHECK_POINTER(this->handle_managers[rid]);

    retval = this->handle_managers[rid]->reallocate_single_handle(ckpt_file, hid, &restored_handle);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore single handle from file: rid(%u), hid(%lu), ckpt_file(%s), retval(%u)",
            rid, hid, ckpt_file.c_str(), retval
        );
        goto exit;
    }
    POS_CHECK_POINTER(restored_handle);

exit:
    return retval;
}


pos_retval_t POSClient_CUDA::__reassign_handle_parents(POSHandle* handle){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i, nb_parent_handles;
    pos_resource_typeid_t rid;
    pos_u64id_t hid;
    POSHandle *parent_handle;

    POS_CHECK_POINTER(handle);
    nb_parent_handles = handle->parent_handles_waitlist.size();

    for(i=0; i<nb_parent_handles; i++){
        rid = handle->parent_handles_waitlist[i].first;
        hid = handle->parent_handles_waitlist[i].second;

        POS_CHECK_POINTER(this->handle_managers[rid]);
        parent_handle = this->handle_managers[rid]->get_handle_by_id(hid);
        if(unlikely(parent_handle == nullptr)){
            POS_WARN_C("no parent handle exist, this might be a bug: rid(%u), hid(%lu)", rid, hid);
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        handle->record_parent_handle(parent_handle);
    }

exit:
    return retval;
}


std::set<pos_resource_typeid_t> POSClient_CUDA::__get_resource_idx(){
    return  std::set<pos_resource_typeid_t>({
        kPOS_ResourceTypeId_CUDA_Context,
        kPOS_ResourceTypeId_CUDA_Module,
        kPOS_ResourceTypeId_CUDA_Function,
        kPOS_ResourceTypeId_CUDA_Var,
        kPOS_ResourceTypeId_CUDA_Device,
        kPOS_ResourceTypeId_CUDA_Memory,
        kPOS_ResourceTypeId_CUDA_Stream,
        kPOS_ResourceTypeId_CUDA_Event,
        kPOS_ResourceTypeId_cuBLAS_Context
    });
}


pos_retval_t POSClient_CUDA::tear_down_all_handles(){
    pos_retval_t retval = POS_SUCCESS;
    
    auto __tear_down_all_typed_handles = [&](pos_resource_typeid_t rid) -> pos_retval_t {
        pos_retval_t dirty_retval = POS_SUCCESS, tmp_retval;
        uint64_t i, nb_handles;
        POSHandle *handle;
        POSHandleManager<POSHandle>* hm;

        POS_CHECK_POINTER(hm = this->handle_managers[rid]);

        nb_handles = hm->get_nb_handles();
        for(i=0; i<nb_handles; i++){
            if(likely(nullptr != (handle = hm->get_handle_by_id(i)))){
                tmp_retval = handle->tear_down();
                if(unlikely(POS_SUCCESS != tmp_retval)){
                    POS_WARN_C(
                        "failed to tear down handle: rid(%u), hid(%lu), retval(%u)",
                        rid, handle->id, tmp_retval
                    );
                    dirty_retval = tmp_retval;
                    continue;
                }
            }
        }
        return dirty_retval;
    };

    /*!
     *  \note   Hierarchy of CUDA Resources
         ╔══════════════════════════════════════════════════════════════════════╗
        ╔══════════════════════════════════════════════════════════════════════╗║
        ║                              CUDA Device                             ║║
        ╠══════════════════════════════════════════════════════════════════════╣║
        ║                             CUDA Context                             ║║
        ╠════════════════╦════════════╦══════════════════════════╦═════════════╣║
        ║   CUDA Stream  ║            ║        CUDA Module       ║             ║║
        ╠════════════════╣ CUDA Event ╠═══════════════╦══════════╣ CUDA Memory ║║
        ║ cuBLAS Context ║            ║ CUDA Function ║ CUDA Var ║             ║╝
        ╚════════════════╩════════════╩═══════════════╩══════════╩═════════════╝
     */

    retval = __tear_down_all_typed_handles(kPOS_ResourceTypeId_CUDA_Memory);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to tear down CUDA memories");
    }

    retval = __tear_down_all_typed_handles(kPOS_ResourceTypeId_CUDA_Event);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to tear down CUDA events");
    }

    retval = __tear_down_all_typed_handles(kPOS_ResourceTypeId_cuBLAS_Context);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to tear down cuBLAS contexts");
    }

    retval = __tear_down_all_typed_handles(kPOS_ResourceTypeId_CUDA_Stream);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to tear down CUDA streams");
    }

    retval = __tear_down_all_typed_handles(kPOS_ResourceTypeId_CUDA_Module);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to tear down CUDA modules");
    }

    // Omit: 
    // CUDA function:   which are deleted when delete CUDA module
    // CUDA var:        which are deleted when delete CUDA module
    // CUDA context:    which are global shared by multiple client
    // CUDA device:     which are global shared by multiple client

exit:
    return retval;
}


void POSClient_CUDA::__dump_hm_cuda_functions() {
    uint64_t nb_functions, i;
    POSHandleManager_CUDA_Function *hm_function;
    POSHandle_CUDA_Function *function_handle;
    std::ofstream output_file;
    std::string dump_content;

    auto dump_function_metas = [](POSHandle_CUDA_Function* function_handle) -> std::string {
        std::string output_str("");
        std::string delimiter("|");
        uint64_t i;
        
        POS_CHECK_POINTER(function_handle);

        // mangled name of the kernel
        output_str += function_handle->name + std::string(delimiter);
        
        // signature of the kernel
        output_str += function_handle->signature + std::string(delimiter);

        // number of paramters
        output_str += std::to_string(function_handle->nb_params);
        output_str += std::string(delimiter);

        // parameter offsets
        for(i=0; i<function_handle->nb_params; i++){
            output_str += std::to_string(function_handle->param_offsets[i]);
            output_str += std::string(delimiter);
        }

        // parameter sizes
        for(i=0; i<function_handle->nb_params; i++){
            output_str += std::to_string(function_handle->param_sizes[i]);
            output_str += std::string(delimiter);
        }

        // input paramters
        output_str += std::to_string(function_handle->input_pointer_params.size());
        output_str += std::string(delimiter);
        for(i=0; i<function_handle->input_pointer_params.size(); i++){
            output_str += std::to_string(function_handle->input_pointer_params[i]);
            output_str += std::string(delimiter);
        }

        // output paramters
        output_str += std::to_string(function_handle->output_pointer_params.size());
        output_str += std::string(delimiter);
        for(i=0; i<function_handle->output_pointer_params.size(); i++){
            output_str += std::to_string(function_handle->output_pointer_params[i]);
            output_str += std::string(delimiter);
        }

        // inout parameters
        output_str += std::to_string(function_handle->inout_pointer_params.size());
        output_str += std::string(delimiter);
        for(i=0; i<function_handle->inout_pointer_params.size(); i++){
            output_str += std::to_string(function_handle->inout_pointer_params[i]);
            output_str += std::string(delimiter);
        }

        // suspicious paramters
        output_str += std::to_string(function_handle->suspicious_params.size());
        output_str += std::string(delimiter);
        for(i=0; i<function_handle->suspicious_params.size(); i++){
            output_str += std::to_string(function_handle->suspicious_params[i]);
            output_str += std::string(delimiter);
        }

        // has verified suspicious paramters
        if(function_handle->has_verified_params){
            output_str += std::string("1") + std::string(delimiter);

            // inout paramters
            output_str += std::to_string(function_handle->confirmed_suspicious_params.size());
            output_str += std::string(delimiter);
            for(i=0; i<function_handle->confirmed_suspicious_params.size(); i++){
                output_str += std::to_string(function_handle->confirmed_suspicious_params[i].first);    // param_index
                output_str += std::string(delimiter);
                output_str += std::to_string(function_handle->confirmed_suspicious_params[i].second);   // offset
                output_str += std::string(delimiter);
            }
        } else {
            output_str += std::string("0") + std::string(delimiter);
        }

        // cbank parameters
        output_str += std::to_string(function_handle->cbank_param_size);

        return output_str;
    };

    // if we have already save the kernels, we can skip
    // if(likely(this->_cxt.is_load_kernel_from_cache == true)){
    //     goto exit;
    // }

    hm_function 
        = (POSHandleManager_CUDA_Function*)(this->handle_managers[kPOS_ResourceTypeId_CUDA_Function]);
    POS_CHECK_POINTER(hm_function);

    output_file.open(this->_cxt.kernel_meta_path.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);

    nb_functions = hm_function->get_nb_handles();
    for(i=0; i<nb_functions; i++){
        POS_CHECK_POINTER(function_handle = hm_function->get_handle_by_id(i));
        output_file << dump_function_metas(function_handle) << std::endl;
    }

    output_file.flush();
    output_file.close();
    POS_LOG("finish dump kernel metadata to %s", this->_cxt.kernel_meta_path.c_str());

exit:
    ;
}
