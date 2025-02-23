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

#include "pos/include/common.h"
#include "pos/include/client.h"
#include "pos/cuda_impl/worker.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace wk_functions {


/*!
 *  \related    cudaMalloc
 *  \brief      allocate a memory area
 */
namespace cuda_malloc {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        CUmemAllocationProp prop = {};
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;
        POSHandle_CUDA_Device *device_handle;
        size_t allocate_size;
        void *ptr;
        CUmemGenericAllocationHandle hdl;
        CUmemAccessDesc access_desc;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        memory_handle = pos_api_create_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_handle->id;

        // create physical memory on the device
        wqe->api_cxt->return_code = cuMemCreate(
            /* handle */ &hdl,
            /* size */ memory_handle->state_size,
            /* prop */ &prop,
            /* flags */ 0
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemCreate: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // map the virtual memory space to the physical memory
        wqe->api_cxt->return_code = cuMemMap(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size,
            /* offset */ 0ULL,
            /* handle */ hdl,
            /* flags */ 0ULL
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemMap: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // set access attribute of this memory
        access_desc.location = prop.location;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        wqe->api_cxt->return_code = cuMemSetAccess(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size,
            /* desc */ &access_desc,
            /* count */ 1ULL
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemSetAccess: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        memory_handle->mark_status(kPOS_HandleStatus_Active);

    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_malloc


/*!
 *  \related    cudaFree
 *  \brief      release a CUDA memory area
 */
namespace cuda_free {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;
        CUmemGenericAllocationHandle hdl;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_delete_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        // obtain the physical memory handle
        wqe->api_cxt->return_code = cuMemRetainAllocationHandle(&hdl, memory_handle->server_addr);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRetainAllocationHandle: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // ummap the virtual memory
        wqe->api_cxt->return_code = cuMemUnmap(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemUnmap: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // release the physical memory
        wqe->api_cxt->return_code = cuMemRelease(hdl);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRelease x 1: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // as we call cuMemRetainAllocationHandle above, we need to release again
        wqe->api_cxt->return_code = cuMemRelease(hdl);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRelease x 2: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        memory_handle->mark_status(kPOS_HandleStatus_Deleted);
        
    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_free


/*!
 *  \related    cudaLaunchKernel
 *  \brief      launch a user-define computation kernel
 */
namespace cuda_launch_kernel {
#define POS_CUDA_LAUNCH_KERNEL_MAX_NB_PARAMS    512

    static void* cuda_args[POS_CUDA_LAUNCH_KERNEL_MAX_NB_PARAMS] = {0};

    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Function *function_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandle *memory_handle;
        uint64_t i, j, nb_involved_memory;
        // void **cuda_args = nullptr;
        void *args, *args_values, *arg_addr;
        uint64_t *addr_list;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        function_handle = (POSHandle_CUDA_Function*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

        stream_handle = (POSHandle_CUDA_Stream*)(pos_api_input_handle(wqe, 1));
        POS_CHECK_POINTER(stream_handle);

        // the 3rd parameter of the API call contains parameter to launch the kernel
        args = pos_api_param_addr(wqe, 3);
        POS_CHECK_POINTER(args);

        // [Cricket Adapt] skip the metadata used by cricket
        args += (sizeof(size_t) + sizeof(uint16_t) * function_handle->nb_params);

        /*!
         *  \note   the actual kernel parameter list passed to the cuLaunchKernel is 
         *          an array of pointers, so we allocate a new array here to store
         *          these pointers
         */
        // TODO: pre-allocated!
        // if(likely(function_handle->nb_params > 0)){
        //     POS_CHECK_POINTER(cuda_args = malloc(function_handle->nb_params * sizeof(void*)));
        // }

        for(i=0; i<function_handle->nb_params; i++){
            cuda_args[i] = args + function_handle->param_offsets[i];
            POS_CHECK_POINTER(cuda_args[i]);
        }
        typedef struct __dim3 { uint32_t x; uint32_t y; uint32_t z; } __dim3_t;

        wqe->api_cxt->return_code = cuLaunchKernel(
            /* f */ (CUfunction)(function_handle->server_addr),
            /* gridDimX */ ((__dim3_t*)pos_api_param_addr(wqe, 1))->x,
            /* gridDimY */ ((__dim3_t*)pos_api_param_addr(wqe, 1))->y,
            /* gridDimZ */ ((__dim3_t*)pos_api_param_addr(wqe, 1))->z,
            /* blockDimX */ ((__dim3_t*)pos_api_param_addr(wqe, 2))->x,
            /* blockDimY */ ((__dim3_t*)pos_api_param_addr(wqe, 2))->y,
            /* blockDimZ */ ((__dim3_t*)pos_api_param_addr(wqe, 2))->z,
            /* sharedMemBytes */ pos_api_param_value(wqe, 4, size_t),
            /* hStream */ (CUstream)(stream_handle->server_addr),
            /* kernelParams */ cuda_args,
            /* extra */ nullptr
        );

        // if(likely(cuda_args != nullptr)){ free(cuda_args); }
        
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_launch_kernel




/*!
 *  \related    cudaMemcpy (Host to Device)
 *  \brief      copy memory buffer from host to device
 */
namespace cuda_memcpy_h2d {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_inout_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize(0);
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpy(
            /* dst */ pos_api_inout_handle_offset_server_addr(wqe, 0),
            /* src */ pos_api_param_addr(wqe, 1),
            /* count */ pos_api_param_size(wqe, 1),
            /* kind */ cudaMemcpyHostToDevice
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_h2d



/*!
 *  \related    cudaMemcpy (Device to Host)
 *  \brief      copy memory buffer from device to host
 */
namespace cuda_memcpy_d2h {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize(0);
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpy(
            /* dst */ wqe->api_cxt->ret_data,
            /* src */ (const void*)(pos_api_input_handle_offset_server_addr(wqe, 0)),
            /* count */ pos_api_param_value(wqe, 1, uint64_t),
            /* kind */ cudaMemcpyDeviceToHost
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
        }
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_d2h




/*!
 *  \related    cudaMemcpy (Device to Device)
 *  \brief      copy memory buffer from device to device
 */
namespace cuda_memcpy_d2d {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *dst_memory_handle, *src_memory_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        dst_memory_handle = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(dst_memory_handle);

        src_memory_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(src_memory_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize(0);
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpy(
            /* dst */ pos_api_output_handle_offset_server_addr(wqe, 0),
            /* src */ pos_api_input_handle_offset_server_addr(wqe, 0),
            /* count */ pos_api_param_value(wqe, 2, uint64_t),
            /* kind */ cudaMemcpyDeviceToDevice
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
        }
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_d2d




/*!
 *  \related    cudaMemcpyAsync (Host to Device)
 *  \brief      async copy memory buffer from host to device
 */
namespace cuda_memcpy_h2d_async {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle, *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_inout_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpyAsync(
            /* dst */ pos_api_inout_handle_offset_server_addr(wqe, 0),
            /* src */ pos_api_param_addr(wqe, 1),
            /* count */ pos_api_param_size(wqe, 1),
            /* kind */ cudaMemcpyHostToDevice,
            /* stream */ (cudaStream_t)(stream_handle->server_addr)
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
        }
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_h2d_async




/*!
 *  \related    cudaMemcpyAsync (Device to Host)
 *  \brief      async copy memory buffer from device to host
 */
namespace cuda_memcpy_d2h_async {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle, *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        stream_handle = pos_api_input_handle(wqe, 1);
        POS_CHECK_POINTER(stream_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpyAsync(
            /* dst */ wqe->api_cxt->ret_data,
            /* src */ pos_api_input_handle_offset_server_addr(wqe, 0),
            /* count */ pos_api_param_value(wqe, 1, uint64_t),
            /* kind */ cudaMemcpyDeviceToHost,
            /* stream */ (cudaStream_t)(stream_handle->server_addr)
        );

        /*! \note   we must synchronize this api under remoting */
        wqe->api_cxt->return_code = cudaStreamSynchronize(
            (cudaStream_t)(stream_handle->server_addr)
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_d2h_async




/*!
 *  \related    cudaMemcpyAsync (Device to Device)
 *  \brief      async copy memory buffer from device to device
 */
namespace cuda_memcpy_d2d_async {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *dst_memory_handle, *src_memory_handle, *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        dst_memory_handle = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(dst_memory_handle);

        src_memory_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(src_memory_handle);

        stream_handle = pos_api_input_handle(wqe, 1);
        POS_CHECK_POINTER(stream_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemcpyAsync(
            /* dst */ pos_api_output_handle_offset_server_addr(wqe, 0),
            /* src */ pos_api_input_handle_offset_server_addr(wqe, 0),
            /* count */ pos_api_param_value(wqe, 2, uint64_t),
            /* kind */ cudaMemcpyDeviceToDevice,
            /* stream */ (cudaStream_t)(stream_handle->server_addr)
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
        }
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_d2d_async


/*!
 *  \related    cudaMemsetAsync
 *  \brief      async set memory area to a specific value
 */
namespace cuda_memset_async {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle, *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        /*!
         *  \note   if we enable overlapped checkpoint, we need to prevent
         *          the checkpoint memcpy conflict with the current memcpy,
         *          so we raise the flag to notify overlapped checkpoint to
         *          provisionally stop
         */
    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = true;
        }
    #endif

        wqe->api_cxt->return_code = cudaMemsetAsync(
            /* devPtr */ pos_api_output_handle_offset_server_addr(wqe, 0),
            /* value */ pos_api_param_value(wqe, 1, int),
            /* count */ pos_api_param_value(wqe, 2, uint64_t),
            /* stream */ (cudaStream_t)(stream_handle->server_addr)
        );

    #if POS_CONF_EVAL_CkptOptLevel == 2
        if( ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.TH_actve == true ){
            wqe->api_cxt->return_code = cudaStreamSynchronize((cudaStream_t)(stream_handle->server_addr));
            if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
                POS_WARN_DETAIL("failed to sync default stream to avoid ckpt conflict")
            }
            ((POSClient*)(wqe->client))->worker->async_ckpt_cxt.membus_lock = false;
        }
    #endif

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_memcpy_d2d_async



/*!
 *  \related    cudaSetDevice
 *  \brief      specify a CUDA device to use
 */
namespace cuda_set_device {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Device *device_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        wqe->api_cxt->return_code = cudaSetDevice(device_handle->id);

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_set_device




/*!
 *  \related    cudaGetLastError
 *  \brief      obtain the latest error within the CUDA context
 */
namespace cuda_get_last_error {
    // parser function
    POS_WK_FUNC_LAUNCH(){
        POS_ERROR_DETAIL("shouldn't be called");
        return POS_SUCCESS;
    }
} // namespace cuda_get_last_error




/*!
 *  \related    cudaGetErrorString
 *  \brief      obtain the error string from the CUDA context
 */
namespace cuda_get_error_string {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        const char* ret_string;
        ret_string = cudaGetErrorString(pos_api_param_value(wqe, 0, cudaError_t));

        if(likely(strlen(ret_string) > 0)){
            memcpy(wqe->api_cxt->ret_data, ret_string, strlen(ret_string)+1);
        }

        wqe->api_cxt->return_code = cudaSuccess;

        POSWorker::__done(ws, wqe);
        
        return POS_SUCCESS;
    }
} // namespace cuda_get_error_string




/*!
 *  \related    cudaPeekAtLastError
 *  \brief      obtain the latest error within the CUDA context
 */
namespace cuda_peek_at_last_error {
    // parser function
    POS_WK_FUNC_LAUNCH(){
        POS_ERROR_DETAIL("shouldn't be called");
        return POS_SUCCESS;
    }
} // namespace cuda_peek_at_last_error




/*!
 *  \related    cudaGetDeviceCount
 *  \brief      obtain the number of devices
 */
namespace cuda_get_device_count {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        // TODO: we launch this op for debug?
        POSWorker::__done(ws, wqe);
        return POS_SUCCESS;
    }
} // namespace cuda_get_device_count





/*!
 *  \related    cudaGetDeviceProperties
 *  \brief      obtain the properties of specified device
 */
namespace cuda_get_device_properties {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Device *device_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        wqe->api_cxt->return_code = cudaGetDeviceProperties(
            (struct cudaDeviceProp*)wqe->api_cxt->ret_data, 
            device_handle->id
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_get_device_properties



/*!
 *  \related    cudaDeviceGetAttribute
 *  \brief      obtain the properties of specified device
 */
namespace cuda_device_get_attribute {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Device *device_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        wqe->api_cxt->return_code = cudaDeviceGetAttribute(
            /* value */ (int*)(wqe->api_cxt->ret_data), 
            /* attr */ pos_api_param_value(wqe, 0, cudaDeviceAttr),
            /* device */ device_handle->id
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_device_get_attribute



/*!
 *  \related    cudaGetDevice
 *  \brief      obtain the handle of specified device
 */
namespace cuda_get_device {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        // TODO: we launch this op for debug?
        POSWorker::__done(ws, wqe);
        return POS_SUCCESS;
    }
} // namespace cuda_get_device



/*!
 *  \related    cudaFuncGetAttributes
 *  \brief      find out attributes for a given function
 */
namespace cuda_func_get_attributes {
    // launch function
    POS_WK_FUNC_LAUNCH(){

        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Function *function_handle;
        struct cudaFuncAttributes *attr = (struct cudaFuncAttributes*)wqe->api_cxt->ret_data;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        function_handle = (POSHandle_CUDA_Function*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

    #define GET_FUNC_ATTR(member, name)					                                    \
        do {								                                                \
            int tmp;								                                        \
            wqe->api_cxt->return_code = cuFuncGetAttribute(                                 \
                &tmp, CU_FUNC_ATTRIBUTE_##name, (CUfunction)(function_handle->server_addr)  \
            );                                                                              \
            if(unlikely(wqe->api_cxt->return_code != CUDA_SUCCESS)){                        \
                goto exit;                                                                  \
            }                                                                               \
            attr->member = tmp;						                                        \
        } while(0)
        GET_FUNC_ATTR(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
        GET_FUNC_ATTR(sharedSizeBytes, SHARED_SIZE_BYTES);
        GET_FUNC_ATTR(constSizeBytes, CONST_SIZE_BYTES);
        GET_FUNC_ATTR(localSizeBytes, LOCAL_SIZE_BYTES);
        GET_FUNC_ATTR(numRegs, NUM_REGS);
        GET_FUNC_ATTR(ptxVersion, PTX_VERSION);
        GET_FUNC_ATTR(binaryVersion, BINARY_VERSION);
        GET_FUNC_ATTR(cacheModeCA, CACHE_MODE_CA);
        GET_FUNC_ATTR(maxDynamicSharedSizeBytes, MAX_DYNAMIC_SHARED_SIZE_BYTES);
        GET_FUNC_ATTR(preferredShmemCarveout, PREFERRED_SHARED_MEMORY_CARVEOUT);
    #undef GET_FUNC_ATTR

    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_func_get_attributes



/*!
 *  \related    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 *  \brief      returns occupancy for a device function with the specified flags
 */
namespace cuda_occupancy_max_active_bpm_with_flags {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Function *function_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        function_handle = (POSHandle_CUDA_Function*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

        wqe->api_cxt->return_code = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            /* numBlocks */ (int*)(wqe->api_cxt->ret_data),
            /* func */ (CUfunction)(function_handle->server_addr),
            /* blockSize */ pos_api_param_value(wqe, 1, int),
            /* dynamicSMemSize */ pos_api_param_value(wqe, 2, size_t),
            /* flags */ pos_api_param_value(wqe, 3, int)
        );

    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_occupancy_max_active_bpm_with_flags




/*!
 *  \related    cudaStreamSynchronize
 *  \brief      sync a specified stream
 */
namespace cuda_stream_synchronize {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        wqe->api_cxt->return_code = cudaStreamSynchronize(
            (cudaStream_t)(stream_handle->server_addr)
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_stream_synchronize




/*!
 *  \related    cudaStreamIsCapturing
 *  \brief      obtain the stream's capturing state
 */
namespace cuda_stream_is_capturing {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        wqe->api_cxt->return_code = cudaStreamIsCapturing(
            /* stream */ (CUstream)(stream_handle->server_addr),
            /* pCaptureStatus */ (cudaStreamCaptureStatus*) wqe->api_cxt->ret_data
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_stream_is_capturing




/*!
 *  \related    cuda_event_create_with_flags
 *  \brief      create a new event with flags
 */
namespace cuda_event_create_with_flags {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *event_handle;
        int flags;
        cudaEvent_t ptr;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);
        
        // execute the actual cudaEventCreateWithFlags
        flags = pos_api_param_value(wqe, 0, int);
        wqe->api_cxt->return_code = cudaEventCreateWithFlags(&ptr, flags);

        // record server address
        if(likely(cudaSuccess == wqe->api_cxt->return_code)){
            event_handle = pos_api_create_handle(wqe, 0);
            POS_CHECK_POINTER(event_handle);
            event_handle->set_server_addr(ptr);
            event_handle->mark_status(kPOS_HandleStatus_Active);
        }

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_event_create_with_flags




/*!
 *  \related    cuda_event_destory
 *  \brief      destory a CUDA event
 */
namespace cuda_event_destory {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *event_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        event_handle = pos_api_delete_handle(wqe, 0);
        POS_CHECK_POINTER(event_handle);

        wqe->api_cxt->return_code = cudaEventDestroy(
            /* event */ (cudaEvent_t)(event_handle->server_addr)
        );

        if(likely(cudaSuccess == wqe->api_cxt->return_code)){
            event_handle->mark_status(kPOS_HandleStatus_Deleted);
        }

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_event_destory




/*!
 *  \related    cuda_event_record
 *  \brief      record a CUDA event
 */
namespace cuda_event_record {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *event_handle, *stream_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        event_handle = pos_api_output_handle(wqe, 0);
        POS_CHECK_POINTER(event_handle);
        stream_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(stream_handle);

        wqe->api_cxt->return_code = cudaEventRecord(
            /* event */ (cudaEvent_t)(event_handle->server_addr),
            /* stream */ (cudaStream_t)(stream_handle->server_addr)
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cuda_event_record



/*!
 *  \related    cudaEventQuery
 *  \brief      query the state of an event
 */
namespace cuda_event_query {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *event_handle;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        event_handle = pos_api_input_handle(wqe, 0);
        POS_CHECK_POINTER(event_handle);

        wqe->api_cxt->return_code = cudaEventQuery(
            /* event */ (cudaEvent_t)(event_handle->server_addr)
        );

        // no need to check state then
        // if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){ 
        //     POSWorker::__restore(ws, wqe);
        // } else {
        //     POSWorker::__done(ws, wqe);
        // }

        POSWorker::__done(ws, wqe);

    exit:
        return retval;
    }
} // namespace cuda_event_query



/*!
 *  \related    template_cuda
 *  \brief      template_cuda
 */
namespace template_cuda {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        return POS_FAILED_NOT_IMPLEMENTED;
    }
} // namespace template_cuda




} // namespace wk_functions 
