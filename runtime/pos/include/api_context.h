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
#include <vector>
#include <map>
#include <string>
#include <memory>

#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/handle.h"
#include "pos/include/utils/timer.h"


// forward declaration
class POSClient;


/*!
 *  \brief  type of api
 */
enum pos_api_type_t : uint8_t {
    /*!
     *  \brief      create XPU sofware/hardware resource
     *  \example    cudaMalloc, cudaCreateStream
     */
    kPOS_API_Type_Create_Resource = 0,

    /*!
     *  \brief      delete XPU sofware/hardware resource
     *  \example    cudaFree, cudaDestoryStream
     */
    kPOS_API_Type_Delete_Resource,

    /*!
     *  \brief      obtain the state of XPU resource
     *  \example    cudaMemcpy (D2H)
     */
    kPOS_API_Type_Get_Resource,

    /*!
     *  \brief      set the state of XPU resource
     *  \example    cudaMemcpy (H2D), cudaLaunchKernel
     */
    kPOS_API_Type_Set_Resource,

    /*!
     *  \brief      checkpoint XPU resource state
     */
    kPOS_API_Type_Checkpoint
};

/*!
 *  \brief  metadata of XPU API
 */
typedef struct POSAPIMeta {
    // whether this is a sync api
    bool is_sync;

    // type of the api
    pos_api_type_t api_type;

    // id of the destination located library (e.g., cuda rt, driver, cublas)
    uint8_t library_id;

    // name of the api
    std::string api_name;
} POSAPIMeta_t;


/*!
 *  \brief  execution type of POSAPIContext_QE
 */
enum pos_apicxt_typeid_t : uint8_t {
    ApiCxt_TypeId_Normal = 0,
    ApiCxt_TypeId_Unexecuted,
    ApiCxt_TypeId_Recomputation
};


/*!
 *  \brief  manager of XPU APIs
 */
class POSApiManager {
 public:
    POSApiManager(){}
    ~POSApiManager() = default;

    /*!
     *  \brief  register metadata of all API on the platform to the manager
     */
    virtual void init(){ POS_ERROR_C("not implemented"); };

    /*!
     *  \brief  translate POS retval to corresponding retval on the XPU platform
     *  \param  pos_retval  the POS retval to be translated
     *  \param  library_id  id of the destination library (e.g., cuda rt, driver, cublas)
     */
    virtual int cast_pos_retval(pos_retval_t pos_retval, uint8_t library_id){ return -1; };

    // map: api_id -> metadata of the api
    std::map<uint64_t, POSAPIMeta_t> api_metas;

 protected:
};


/*!
 *  \brief  execution state of an API instance
 */
enum pos_api_execute_status_t : uint8_t {
    kPOS_API_Execute_Status_Init = 0,
    kPOS_API_Execute_Status_Return_After_Parse,
    kPOS_API_Execute_Status_Return_Without_Worker,
    kPOS_API_Execute_Status_Parser_Failed,
    kPOS_API_Execute_Status_Worker_Failed
};


/*!
 *  \brief  descriptor of one parameter of an API call
 */
typedef struct POSAPIParam {
    // payload of the parameter
    void *param_value;

    // size of the parameter
    size_t param_size;

    /*!
     *  \brief  constructor
     *  \param  src_value   pointer to the actual value of the parameter
     *  \param  size        size of the parameter
     */
    POSAPIParam(void *src_value, size_t size) : param_size(size) {
        POS_CHECK_POINTER(param_value = malloc(size));
        memcpy(param_value, src_value, param_size);
    }

    /*!
     *  \brief  deconstructor
     */
    ~POSAPIParam(){
        POS_CHECK_POINTER(param_value); free(param_value);
    }
} POSAPIParam_t;


/*!
 *  \brief  macro to obtain parameter value of an API instance by given index,
 *          and cast to corresponding type
 */
#define pos_api_param_value(qe_ptr, index, type)                \
    (*((type*)(qe_ptr->api_cxt->params[index]->param_value)))

#define pos_api_param_addr(qe_ptr, index)           \
    (qe_ptr->api_cxt->params[index]->param_value)

#define pos_api_param_size(qe_ptr, index)           \
    (qe_ptr->api_cxt->params[index]->param_size)


/*!
 *  \brief  descriptor of one parameter of an API call
 */
typedef struct POSAPIParamDesp { void *value; size_t size; } POSAPIParamDesp_t;


/*!
 *  \brief  context of an API call
 */
typedef struct POSAPIContext {
    // index of the called API
    uint64_t api_id;

    // parameter list of the called API
    std::vector<POSAPIParam_t*> params;

    // pointer to the area to store the return result
    void *ret_data;
    
    // size of the return value
    uint64_t retval_size;

    // return code of the API
    int return_code;

    /*!
     *  \brief  constructor
     *  \param  api_id_         index of the called API
     *  \param  param_desps     descriptors of all involved parameters
     *  \param  ret_data_       pointer to the memory area that store the returned value
     *  \param  retval_size_    size of the return value
     */
    POSAPIContext(
        uint64_t api_id_, std::vector<POSAPIParamDesp_t>& param_desps, void* ret_data_=nullptr, uint64_t retval_size_=0
    );


    /*!
     *  \brief  constructor
     *  \note   this constructor is for restoring
     *  \param  api_id_     specialized API index of the checkpointing op
     *  \param  retval_size size of return value
     */
    POSAPIContext(uint64_t api_id_, uint64_t retval_size);


    ~POSAPIContext(){
        for(auto param : params){ POS_CHECK_POINTER(param); delete param; }
    }
} POSAPIContext_t;


/*!
 *  \brief  view of the api instance to use the handle
 */
typedef struct POSHandleView {
    // pointer to the used handle
    POSHandle *handle;

    /*!
     *  \brief  id of the handle inside handle manager list
     *  \note   this field is only used during restoring phrase
     */
    pos_u64id_t id;

    /*!
     *  \brief  resource type index of the handle
     *  \note   this field is only used during restoring phrase
     */
    pos_resource_typeid_t resource_type_id;

    /*!
     *  \brief      index of the corresponding parameter of this handle view
     *  \example    for API such as launchKernel, we need to know which parameter this handle 
     *              is corresponding to, in order to translating the address in the worker 
     *              launching function
     */
    uint64_t param_index;

    /*!
     *  \brief      offset from the base address of the handle
     *  \example    for memory handle, the client might provide an non-base address, so we should
     *              record the offset here, so that we can obtain the correct server-side address
     *              within the worker launching function
     */
    uint64_t offset;

    /*!
     *  \brief  constructor
     *  \param  handle_             pointer to the handle which is view targeted on
     *  \param  param_index_        index of the corresponding parameter of this handle view
     *  \param  offset_             offset from the base address of the handle
     */
    POSHandleView(
        POSHandle* handle_, uint64_t param_index_ = 0, uint64_t offset_ = 0
    ) : handle(handle_), param_index(param_index_), offset(offset_){}

    /*!
     *  \brief  constructor
     *  \note   this constructor is used only during restore phrase
     */
    POSHandleView() : handle(nullptr), param_index(0), offset(0){}
} POSHandleView_t;


/*!
 *  \brief  work queue element, as the element within work 
 *          queue between frontend and runtime
 */
typedef struct POSAPIContext_QE {
    // uuid of the remote client
    pos_client_uuid_t client_id;

    // pointer to the POSClient instance
    POSClient *client;

    // uuid of this API call instance within the client
    uint64_t id;

    // context of the called API
    POSAPIContext *api_cxt;
    
    // mark whether current WQE has returned to RPC thread
    bool has_return;

    // execution status of the API call
    pos_api_execute_status_t status;

    // ApiCxt Type
    pos_apicxt_typeid_t type;

    // all involved handle during the processing of this API instance
    std::vector<POSHandleView_t> input_handle_views;
    std::vector<POSHandleView_t> output_handle_views;
    std::vector<POSHandleView_t> create_handle_views;
    std::vector<POSHandleView_t> delete_handle_views;
    std::vector<POSHandleView_t> inout_handle_views;

    /* =========== tracing fields =========== */
    uint64_t create_tick, return_tick;
    uint64_t parser_s_tick, parser_e_tick, worker_s_tick, worker_e_tick;
    /* ======= end of profiling fields ======== */


    /*!
     *  \brief  constructor
     *  \param  api_id          index of the called API
     *  \param  uuid            uuid of the remote client
     *  \param  param_desps     description of all parameters of the call
     *  \param  inst_id         uuid of this API call instance within the client
     *  \param  retval_data     pointer to the memory area that store the returned value
     *  \param  retval_size     size of the return value
     *  \param  pos_client      pointer to the POSClient instance
     */
    POSAPIContext_QE(
        uint64_t api_id, pos_client_uuid_t uuid, std::vector<POSAPIParamDesp_t>& param_desps,
        uint64_t inst_id, void* retval_data, uint64_t retval_size, POSClient* pos_client
    );


    /*!
     *  \brief  constructor
     *  \note   this constructor is for restoring from binary checkpoint file
     *  \param  client      pointer to the POSClient instance
     *  \param  ckpt_file   path to the checkpoint file
     *  \param  type        type of the restored APIContext, either ApiCxt_TypeId_Unexecuted
     *                      or ApiCxt_TypeId_Recomputation
     */
    POSAPIContext_QE(POSClient* client, const std::string& ckpt_file, pos_apicxt_typeid_t type);


    /*!
     *  \brief  deconstructor
     */
    ~POSAPIContext_QE();


    /*!
     *  \brief  persist the state of this APIcontext to specified directory
     *  \tparam with_params whether to persist with parameter information,
     *          if this persist is for tracing, then false; otherwise for
     *          checkpointing, then true   
     *  \tparam type        type of the persist wqe:
     *                      1: unexecuted APIs
     *                      2: recomputation APIs
     *  \param  ckpt_dir    directory to store the checkpoint
     *  \return POS_SUCCESS for successfully checkpointing
     */
    template<bool with_params, pos_apicxt_typeid_t type>
    pos_retval_t persist(std::string ckpt_dir);


    /*!
     *  \brief  record involved handles of this API instance
     *  \param  handle_view     view of the API instance to use this handle
     */
    template<pos_edge_direction_t dir>
    inline void record_handle(POSHandleView_t&& handle_view){
        if constexpr (dir == kPOS_Edge_Direction_In){
            input_handle_views.emplace_back(handle_view);
        } else if (dir == kPOS_Edge_Direction_Out){
            output_handle_views.emplace_back(handle_view);
        } else if (dir == kPOS_Edge_Direction_Create){
            create_handle_views.emplace_back(handle_view);
        } else if (dir == kPOS_Edge_Direction_Delete){
            delete_handle_views.emplace_back(handle_view);
        } else { // inout
            inout_handle_views.emplace_back(handle_view);
        }
    }
} POSAPIContext_QE_t;


#define pos_api_output_handle_view(qe_ptr, index)           \
    (qe_ptr->output_handle_views[index])

#define pos_api_input_handle_view(qe_ptr, index)            \
    (qe_ptr->input_handle_views[index])

#define pos_api_create_handle_view(qe_ptr, index)           \
    (qe_ptr->create_handle_views[index])

#define pos_api_delete_handle_view(qe_ptr, index)           \
    (qe_ptr->delete_handle_views[index])

#define pos_api_inout_handle_view(qe_ptr, index)            \
    (qe_ptr->inout_handle_views[index])


#define pos_api_output_handle(qe_ptr, index)                \
    (qe_ptr->output_handle_views[index].handle)

#define pos_api_input_handle(qe_ptr, index)                 \
    (qe_ptr->input_handle_views[index].handle)

#define pos_api_create_handle(qe_ptr, index)                \
    (qe_ptr->create_handle_views[index].handle)

#define pos_api_delete_handle(qe_ptr, index)                \
    (qe_ptr->delete_handle_views[index].handle)

#define pos_api_inout_handle(qe_ptr, index)                 \
    (qe_ptr->inout_handle_views[index].handle)


#define pos_api_input_handle_offset_server_addr(qe_ptr, index)  \
    ((void*)((uint64_t)(qe_ptr->input_handle_views[index].handle->server_addr) + (qe_ptr->input_handle_views[index].offset)))

#define pos_api_output_handle_offset_server_addr(qe_ptr, index)  \
    ((void*)((uint64_t)(qe_ptr->output_handle_views[index].handle->server_addr) + (qe_ptr->output_handle_views[index].offset)))

#define pos_api_inout_handle_offset_server_addr(qe_ptr, index)  \
    ((void*)((uint64_t)(qe_ptr->inout_handle_views[index].handle->server_addr) + (qe_ptr->inout_handle_views[index].offset)))
