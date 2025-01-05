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
#include <map>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/log.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/handle/memory.h"
#include "pos/cuda_impl/proto/memory.pb.h"


std::map<int, CUdeviceptr>  POSHandleManager_CUDA_Memory::alloc_ptrs;
std::map<int, uint64_t>     POSHandleManager_CUDA_Memory::alloc_granularities;
bool                        POSHandleManager_CUDA_Memory::has_finshed_reserved;
const uint64_t              POSHandleManager_CUDA_Memory::reserved_vm_base = 0x7facd0000000;


POSHandle_CUDA_Memory::POSHandle_CUDA_Memory(size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(size_, hm, id_, state_size_)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Memory;

#if POS_CONF_EVAL_CkptOptLevel > 0 || POS_CONF_EVAL_MigrOptLevel > 0
    // initialize checkpoint bag
    if(unlikely(POS_SUCCESS != this->__init_ckpt_bag())){
        POS_ERROR_C_DETAIL("failed to inilialize checkpoint bag");
    }
#endif
}


POSHandle_CUDA_Memory::POSHandle_CUDA_Memory(void* hm) : POSHandle_CUDA(hm)
{
    this->resource_type_id = kPOS_ResourceTypeId_CUDA_Memory;
}


POSHandle_CUDA_Memory::POSHandle_CUDA_Memory(void *client_addr_, size_t size_, void* hm, pos_u64id_t id_, size_t state_size_)
    : POSHandle_CUDA(client_addr_, size_, hm, id_, state_size_)
{
    POS_ERROR_C_DETAIL("shouldn't be called");
}


pos_retval_t POSHandle_CUDA_Memory::tear_down(){
    pos_retval_t retval = POS_SUCCESS;
    CUresult cuda_dv_retval;
    CUmemGenericAllocationHandle hdl;

    if(unlikely(this->status != kPOS_HandleStatus_Active)){ goto exit; }

    // obtain the physical memory handle
    cuda_dv_retval = cuMemRetainAllocationHandle(&hdl, this->server_addr);
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_DETAIL(
            "failed to tear down CUDA memory, failed to call cuMemRetainAllocationHandle: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

    // ummap the virtual memory
    cuda_dv_retval = cuMemUnmap(
        /* ptr */ (CUdeviceptr)(this->server_addr),
        /* size */ this->state_size
    );
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_DETAIL(
            "failed to tear down CUDA memory, failed to call cuMemUnmap: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

    // release the physical memory
    cuda_dv_retval = cuMemRelease(hdl);
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_DETAIL(
            "failed to tear down CUDA memory, failed to call cuMemRelease x 1: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

    // as we call cuMemRetainAllocationHandle above, we need to release again
    cuda_dv_retval = cuMemRelease(hdl);
    if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
        POS_WARN_DETAIL(
            "failed to tear down CUDA memory, failed to call cuMemRelease x 2: id(%lu), client_addr(%p), server_addr(%p), retval(%d)",
            this->id, this->client_addr, this->server_addr, cuda_dv_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Memory::__init_ckpt_bag(){ 
    this->ckpt_bag = new POSCheckpointBag(
        this->state_size,
        this->__checkpoint_allocator,
        this->__checkpoint_deallocator,
        this->__checkpoint_dev_allocator,
        this->__checkpoint_dev_deallocator
    );
    POS_CHECK_POINTER(this->ckpt_bag);
    return POS_SUCCESS;
}


pos_retval_t POSHandle_CUDA_Memory::__add(uint64_t version_id, uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;
    POSCheckpointSlot* ckpt_slot;

    // apply new on-device checkpoint slot
    if(unlikely(POS_SUCCESS != (
        this->ckpt_bag->template apply_checkpoint_slot<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
            /* version */ version_id,
            /* ptr */ &ckpt_slot,
            /* dynamic_state_size */ 0,
            /* force_overwrite */ true
        )
    ))){
        POS_WARN_C("failed to apply checkpoint slot");
        retval = POS_FAILED;
        goto exit;
    }

    cuda_rt_retval = cudaMemcpyAsync(
        /* dst */ ckpt_slot->expose_pointer(), 
        /* src */ this->server_addr,
        /* size */ this->state_size,
        /* kind */ cudaMemcpyDeviceToDevice,
        /* stream */ (cudaStream_t)(stream_id)
    );
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C(
            "failed to checkpoint memory handle on device: server_addr(%p), retval(%d)",
            this->server_addr, cuda_rt_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

    cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C(
            "failed to synchronize after checkpointing memory handle on device: server_addr(%p), retval(%d)",
            this->server_addr, cuda_rt_retval
        );
        retval = POS_FAILED;
        goto exit;
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Memory::__commit(uint64_t version_id, uint64_t stream_id, bool from_cache, bool is_sync){ 
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;
    POSCheckpointSlot *ckpt_slot, *cow_ckpt_slot;
    
    // TODO: [zhuobin] why we have this call??
    cudaSetDevice(0);

    // apply new host-side checkpoint slot for device-side state
    if(unlikely(POS_SUCCESS != (
        this->ckpt_bag->template apply_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
            /* version */ version_id,
            /* ptr */ &ckpt_slot,
            /* dynamic_state_size */ 0,
            /* force_overwrite */ true
        )
    ))){
        POS_WARN_C("failed to apply host-side checkpoint slot");
        retval = POS_FAILED;
        goto exit;
    }

    POS_CHECK_POINTER(ckpt_slot);

    if(from_cache == false){
        // commit from origin buffer
        cuda_rt_retval = cudaMemcpyAsync(
            /* dst */ ckpt_slot->expose_pointer(), 
            /* src */ this->server_addr,
            /* size */ this->state_size,
            /* kind */ cudaMemcpyDeviceToHost,
            /* stream */ (cudaStream_t)(stream_id)
        );
        if(unlikely(cuda_rt_retval != cudaSuccess)){
            POS_WARN_C(
                "failed to checkpoint memory handle from origin buffer: server_addr(%p), retval(%d)",
                this->server_addr, cuda_rt_retval
            );
            retval = POS_FAILED;
            goto exit;
        }
    } else {
        // commit from cache buffer
        if(unlikely(POS_SUCCESS != (
            this->ckpt_bag->template get_checkpoint_slot<kPOS_CkptSlotPosition_Device, kPOS_CkptStateType_Device>(
                /* ptr */ &cow_ckpt_slot,
                /* version */ version_id
            )
        ))){
            POS_ERROR_C_DETAIL(
                "no cache buffer with the version founded, this is a bug: version_id(%lu), server_addr(%p)",
                version_id, this->server_addr
            );
        }
        cuda_rt_retval = cudaMemcpyAsync(
            /* dst */ ckpt_slot->expose_pointer(), 
            /* src */ cow_ckpt_slot->expose_pointer(),
            /* size */ this->state_size,
            /* kind */ cudaMemcpyDeviceToHost,
            /* stream */ (cudaStream_t)(stream_id)
        );
        if(unlikely(cuda_rt_retval != cudaSuccess)){
            POS_WARN_C(
                "failed to checkpoint memory handle from COW buffer: server_addr(%p), retval(%d)",
                this->server_addr, cuda_rt_retval
            );
            retval = POS_FAILED;
            goto exit;
        }
    }

    if(is_sync){
        cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
        if(unlikely(cuda_rt_retval != cudaSuccess)){
            POS_WARN_C(
                "failed to synchronize after commiting memory handle: server_addr(%p), retval(%d)",
                this->server_addr, cuda_rt_retval
            );
            retval = POS_FAILED;
            goto exit;
        }
    }

exit:
    return retval;
}


pos_retval_t POSHandle_CUDA_Memory::__get_checkpoint_slot_for_persist(POSCheckpointSlot** ckpt_slot, uint64_t version_id){
    pos_retval_t retval = POS_SUCCESS;

    POS_CHECK_POINTER(ckpt_slot);

    if(unlikely(POS_SUCCESS != (
        retval = this->ckpt_bag->template get_checkpoint_slot<kPOS_CkptSlotPosition_Host, kPOS_CkptStateType_Device>(
            /* ckpt_slot */ ckpt_slot,
            /* version */ version_id
        )
    ))){
        POS_WARN_C("failed to obtain checkpoint slot for persist: version_id(%lu), retval(%d)", version_id, retval);
        goto exit;
    }

exit:
    return retval;
}



pos_retval_t POSHandle_CUDA_Memory::__generate_protobuf_binary(google::protobuf::Message** binary, google::protobuf::Message** base_binary){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Memory *cuda_memory_binary;

    POS_CHECK_POINTER(binary);
    POS_CHECK_POINTER(base_binary);

    cuda_memory_binary = new pos_protobuf::Bin_POSHandle_CUDA_Memory();
    POS_CHECK_POINTER(cuda_memory_binary);

    *binary = reinterpret_cast<google::protobuf::Message*>(cuda_memory_binary);
    POS_CHECK_POINTER(*binary);
    *base_binary = cuda_memory_binary->mutable_base();
    POS_CHECK_POINTER(*base_binary);

    // serialize handle specific fields
    /* currently nothing */

    return retval;
}


pos_retval_t POSHandle_CUDA_Memory::__restore(){
    pos_retval_t retval = POS_SUCCESS;

    cudaError_t cuda_rt_retval;
    CUresult cuda_dv_retval;
    POSHandle_CUDA_Device *device_handle;
    
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle hdl;
    CUmemAccessDesc access_desc;

    void *rt_ptr;

    POS_ASSERT(this->parent_handles.size() == 1);
    POS_CHECK_POINTER(device_handle = static_cast<POSHandle_CUDA_Device*>(this->parent_handles[0]));

    if(likely(this->server_addr != 0)){
        /*!
            *  \note   case:   restore memory handle at the specified memory address
            */
        POS_ASSERT(this->client_addr == this->server_addr);

        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_handle->id;

        cuda_dv_retval = cuMemCreate(
            /* handle */ &hdl,
            /* size */ this->state_size,
            /* prop */ &prop,
            /* flags */ 0
        );
        if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
            POS_WARN_DETAIL(
                "failed to execute cuMemCreate while restoring: client_addr(%p), state_size(%lu), retval(%d)",
                this->client_addr, this->state_size, cuda_dv_retval
            );
            retval = POS_FAILED;
            goto exit;
        }

        cuda_dv_retval = cuMemMap(
            /* ptr */ (CUdeviceptr)(this->server_addr),
            /* size */ this->state_size,
            /* offset */ 0ULL,
            /* handle */ hdl,
            /* flags */ 0ULL
        );
        if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
            POS_WARN_DETAIL(
                "failed to execute cuMemMap while restoring: client_addr(%p), state_size(%lu), retval(%d)",
                this->client_addr, this->state_size, cuda_dv_retval
            );
            retval = POS_FAILED;
            goto exit;
        }

        // set access attribute of this memory
        access_desc.location = prop.location;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuda_dv_retval = cuMemSetAccess(
            /* ptr */ (CUdeviceptr)(this->server_addr),
            /* size */ this->state_size,
            /* desc */ &access_desc,
            /* count */ 1ULL
        );
        if(unlikely(CUDA_SUCCESS != cuda_dv_retval)){
            POS_WARN_DETAIL(
                "failed to execute cuMemSetAccess while restoring: client_addr(%p), state_size(%lu), retval(%d)",
                this->client_addr, this->state_size, cuda_dv_retval
            );
            retval = POS_FAILED;
            goto exit;
        }
    } else {
        /*!
         *  \note   case:   no specified address to restore, randomly assign one
         */
        cuda_rt_retval = cudaMalloc(&rt_ptr, this->state_size);
        if(unlikely(cuda_rt_retval != cudaSuccess)){
            retval = POS_FAILED;
            POS_WARN_C_DETAIL("failed to restore CUDA memory, cudaMalloc failed: %d", cuda_rt_retval);
            goto exit;
        }

        retval = this->set_passthrough_addr(rt_ptr, this);
        if(unlikely(POS_SUCCESS != retval)){ 
            POS_WARN_DETAIL("failed to restore CUDA memory, failed to set passthrough address for the memory handle: %p", rt_ptr);
            goto exit;
        }
    }

    this->mark_status(kPOS_HandleStatus_Active);

exit:
    return retval;
}



pos_retval_t POSHandle_CUDA_Memory::__reload_state(void* mapped, uint64_t ckpt_file_size, uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Memory memory_binary;
    cudaError_t cuda_rt_retval;

    POS_CHECK_POINTER(mapped);

    if(!memory_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle state, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(memory_binary.mutable_base());

    #if POS_CONF_RUNTIME_EnableTrace
        ((POSHandleManager_CUDA_Memory*)(this->_hm))->metric_tickers.start(POSHandleManager_CUDA_Memory::RESTORE_reload_state);
    #endif

    cuda_rt_retval = cudaMemcpyAsync(
        /* dst */ this->server_addr,
        /* src */ reinterpret_cast<const void*>(memory_binary.mutable_base()->state().c_str()),
        /* count */ this->state_size,
        /* kind */ cudaMemcpyHostToDevice,
        /* stream */ (cudaStream_t)(stream_id)
    );
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_DETAIL("failed to reload state of CUDA memory: server_addr(%p), retval(%d)", this->server_addr, cuda_rt_retval);
        retval = POS_FAILED;
        goto exit;
    }

    cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_DETAIL("failed to synchronize after reloading state of CUDA memory: server_addr(%p), retval(%d)", this->server_addr, cuda_rt_retval);
        retval = POS_FAILED;
        goto exit;
    }

    #if POS_CONF_RUNTIME_EnableTrace
        ((POSHandleManager_CUDA_Memory*)(this->_hm))->metric_tickers.end(POSHandleManager_CUDA_Memory::RESTORE_reload_state);
    #endif

exit:
    // this should be the end of using this mmap area, so we release it here
    munmap(mapped, ckpt_file_size);
    return retval;
}


POSHandleManager_CUDA_Memory::POSHandleManager_CUDA_Memory() : POSHandleManager(/* passthrough */ true) {}


pos_retval_t POSHandleManager_CUDA_Memory::init(std::map<uint64_t, std::vector<POSHandle*>> related_handles, bool is_restoring){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t nb_context, i, j;
    POSHandle *context_handle;
    
    this->_rid = kPOS_ResourceTypeId_CUDA_Memory;

    /*!
     *  \brief  reserve a large portion of virtual memory space on a specified device
     *  \param  context_handle  handle of the context of the specified device
     */
    auto __reserve_device_vm_space = [](POSHandle *context_handle) -> pos_retval_t {
        uint64_t free_portion, free, total;
        uint64_t reserved_size, alloc_granularity;
        CUmemAllocationProp prop = {};
        CUmemGenericAllocationHandle hdl;
        CUmemAccessDesc accessDesc;
        CUdeviceptr ptr;
        pos_retval_t retval = POS_SUCCESS, tmp_retval;
        cudaError_t rt_retval;
        CUresult dv_retval;
        CUcontext old_ctx;
        bool do_ctx_switch = false;
        int device_id;

        POS_ASSERT(context_handle->parent_handles.size() == 1);
        POS_ASSERT(context_handle->parent_handles[0]->resource_type_id == kPOS_ResourceTypeId_CUDA_Device);
        device_id = static_cast<int>((uint64_t)(context_handle->parent_handles[0]->client_addr));

        POS_ASSERT(POSHandleManager_CUDA_Memory::alloc_ptrs.count(device_id) == 0);
        POS_ASSERT(POSHandleManager_CUDA_Memory::alloc_granularities.count(device_id) == 0);

        // switch to target device
        if(unlikely(CUDA_SUCCESS != (
            dv_retval = cuCtxPushCurrent(static_cast<CUcontext>(context_handle->server_addr))
        ))){
            POS_WARN(
                "failed to preserve memory on CUDA device, failed to call cuCtxPushCurrent: retval(%d), device_id(%d)",
                dv_retval, device_id
            );
            retval = POS_FAILED_DRIVER;
            goto exit;
        }
        cuCtxSynchronize();
        do_ctx_switch = true;

        // obtain avaliable device memory space
        rt_retval = cudaMemGetInfo(&free, &total);
        if(unlikely(rt_retval == cudaErrorMemoryAllocation || free < 16*1024*1024)){
            POS_LOG("no available memory space on device to reserve, skip: device_id(%d)", device_id);
            POSHandleManager_CUDA_Memory::alloc_granularities[device_id] = 0;
            POSHandleManager_CUDA_Memory::alloc_ptrs[device_id] = (CUdeviceptr)(nullptr);
            goto exit;
        }
        if(unlikely(cudaSuccess != rt_retval)){
            POS_WARN("failed to call cudaMemGetInfo: retval(%d)", rt_retval);
        }

        // obtain granularity of allocation
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        if(unlikely(CUDA_SUCCESS != (
            dv_retval = cuMemGetAllocationGranularity(&alloc_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        ))){
            POS_WARN(
                "failed to preserve memory on CUDA device, failed to call cuMemGetAllocationGranularity: retval(%d), device_id(%d)",
                dv_retval, device_id
            );
            retval = POS_FAILED_DRIVER;
            goto exit;
        }
        POSHandleManager_CUDA_Memory::alloc_granularities[device_id] = alloc_granularity;

        /*!
         *  \note   we only reserved 90% of free memory, and round up the size according to allocation granularity
         */
    #define ROUND_UP(size, aligned_size) ((size + aligned_size - 1) / aligned_size) * aligned_size
        free_portion = 0.9*free;
        reserved_size = ROUND_UP(free_portion, alloc_granularity);
    #undef ROUND_UP

        if(unlikely(CUDA_SUCCESS != (
            dv_retval = cuMemAddressReserve(&ptr, reserved_size, 0, POSHandleManager_CUDA_Memory::reserved_vm_base, 0ULL)
        ))){
            POS_WARN(
                "failed to preserve memory on CUDA device, failed to call cuMemAddressReserve: retval(%d), device_id(%d)",
                dv_retval, device_id
            );
            if(likely(dv_retval == CUDA_ERROR_OUT_OF_MEMORY)){
                retval = POS_FAILED_OOM;
            } else {
                retval = POS_FAILED_DRIVER;
            }

            goto exit;
        }
        POSHandleManager_CUDA_Memory::alloc_ptrs[device_id] = ptr;
        POS_LOG("reserved virtual memory space: device_id(%d), base(%p), size(%lu)", device_id, ptr, reserved_size);

    exit:
        if(do_ctx_switch == true){
            // switch back to old context
            if(unlikely(CUDA_SUCCESS != (
                dv_retval = cuCtxPopCurrent(&old_ctx)
            ))){
                POS_WARN("preserved memory on CUDA device, but failed to call cuCtxPopCurrent: retval(%d)", dv_retval);
                retval = POS_FAILED_DRIVER;
            } else {
                cuCtxSynchronize();
            }
        }
        return retval;
    };

    if(unlikely(related_handles.count(kPOS_ResourceTypeId_CUDA_Context) == 0)){
        retval = POS_FAILED_INVALID_INPUT;
        POS_WARN_C("failed to init handle manager for CUDA memory, no context provided");
        goto exit;
    }

    nb_context = related_handles[kPOS_ResourceTypeId_CUDA_Context].size();
    if(unlikely(nb_context == 0)){
        retval = POS_FAILED_INVALID_INPUT;
        POS_WARN_C("failed to init handle manager for CUDA memory, no context provided");
        goto exit;
    }

    // no need to conduct reserving if previous hm has already done
    if(this->has_finshed_reserved == true){ goto exit; }

    // we reserve virtual memory space on each device
    for(i=0; i<nb_context; i++){
        POS_CHECK_POINTER(context_handle = related_handles[kPOS_ResourceTypeId_CUDA_Context][i]);
        if(unlikely(POS_SUCCESS != (
            retval = __reserve_device_vm_space(context_handle)
        ))){
            if(retval == POS_FAILED_OOM){
                POS_WARN_C(
                    "failed to preserve memory space on device, out of memory, omit this device",
                    context_handle->client_addr
                );
                retval = POS_SUCCESS;
            } else {
                POS_WARN_C(
                    "failed to preserve memory space on device: context_client_addr(%p), retval(%u)",
                    context_handle->client_addr, retval
                );
            }
        }
    }

    this->has_finshed_reserved = true;

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Memory::allocate_mocked_resource(
    POSHandle_CUDA_Memory** handle,
    std::map<uint64_t, std::vector<POSHandle*>> related_handles,
    size_t size,
    bool use_expected_addr,
    uint64_t expected_addr,
    uint64_t state_size
){
    pos_retval_t retval = POS_SUCCESS;
    CUdeviceptr alloc_ptr;
    uint64_t aligned_alloc_size;
    POSHandle *context_handle, *device_handle;
    int device_id;

    POS_CHECK_POINTER(handle);

    // get parent context handle
    POS_ASSERT(related_handles.count(kPOS_ResourceTypeId_CUDA_Context) == 1);
    POS_ASSERT(related_handles[kPOS_ResourceTypeId_CUDA_Context].size() == 1);
    POS_CHECK_POINTER(context_handle = related_handles[kPOS_ResourceTypeId_CUDA_Context][0]);
    
    // get device id based on context handle
    POS_ASSERT(context_handle->parent_handles.size() == 1);
    POS_CHECK_POINTER(device_handle = context_handle->parent_handles[0]);
    POS_ASSERT(device_handle->resource_type_id == kPOS_ResourceTypeId_CUDA_Device);
    device_id = static_cast<int>((uint64_t)(device_handle->client_addr));

    POS_ASSERT(POSHandleManager_CUDA_Memory::alloc_ptrs.count(device_id) == 1);
    POS_ASSERT(POSHandleManager_CUDA_Memory::alloc_granularities.count(device_id) == 1);

    // obtain the desired address based on reserved virtual memory space pointer
    alloc_ptr = POSHandleManager_CUDA_Memory::alloc_ptrs[device_id];

    // no avaialble memory space on device
    if(unlikely((void*)(alloc_ptr) == nullptr)){
        retval = POS_FAILED_DRAIN;
        goto exit;
    }

    // forward the allocation pointer
#define ROUND_UP(size, alloc_granularity) ((size + alloc_granularity - 1) / alloc_granularity) * alloc_granularity
    aligned_alloc_size = ROUND_UP(state_size, POSHandleManager_CUDA_Memory::alloc_granularities[device_id]);
    POSHandleManager_CUDA_Memory::alloc_ptrs[device_id] += aligned_alloc_size;
#undef ROUND_UP

    retval = this->__allocate_mocked_resource(handle, size, use_expected_addr, expected_addr, aligned_alloc_size);
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C("failed to allocate mocked CUDA memory in the manager");
        goto exit;
    }

    POS_CHECK_POINTER(*handle);
    (*handle)->record_parent_handle(context_handle);

    // we directly setup the passthrough address here
    (*handle)->set_passthrough_addr((void*)(alloc_ptr), (*handle));

exit:
    return retval;
}


pos_retval_t POSHandleManager_CUDA_Memory::preserve_pooled_handles(uint64_t amount){
    return POS_SUCCESS;
}


pos_retval_t POSHandleManager_CUDA_Memory::try_restore_from_pool(POSHandle_CUDA_Memory* handle){
    return POS_FAILED;
}


pos_retval_t POSHandleManager_CUDA_Memory::__reallocate_single_handle(void* mapped, uint64_t ckpt_file_size, POSHandle_CUDA_Memory** handle){
    pos_retval_t retval = POS_SUCCESS;
    pos_protobuf::Bin_POSHandle_CUDA_Memory cuda_memory_binary;
    int i, nb_parent_handles, nb_parent_handles_;
    std::vector<std::pair<pos_resource_typeid_t, pos_u64id_t>> parent_handles_waitlist;
    pos_resource_typeid_t parent_handle_rid;
    pos_u64id_t parent_handle_hid;

    POS_CHECK_POINTER(mapped);
    POS_CHECK_POINTER(handle);

    if(!cuda_memory_binary.ParseFromArray(mapped, ckpt_file_size)){
        POS_WARN_C("failed to restore handle, failed to deserialize from mmap area");
        retval = POS_FAILED;
        goto exit;
    }
    POS_CHECK_POINTER(cuda_memory_binary.mutable_base());

    // form parent handles waitlist
    nb_parent_handles = cuda_memory_binary.mutable_base()->parent_handle_resource_type_idx_size();
    nb_parent_handles_ = cuda_memory_binary.mutable_base()->parent_handle_idx_size();
    POS_ASSERT(nb_parent_handles == nb_parent_handles_);
    for (i=0; i<nb_parent_handles; i++) {
        parent_handle_rid = cuda_memory_binary.mutable_base()->parent_handle_resource_type_idx(i);
        parent_handle_hid = cuda_memory_binary.mutable_base()->parent_handle_idx(i);
        parent_handles_waitlist.push_back({ parent_handle_rid, parent_handle_hid });
    }

    // create resource shell in this handle manager
    retval = this->__restore_mocked_resource(
        /* handle */ handle,
        /* id */ cuda_memory_binary.mutable_base()->id(),
        /* client_addr */ cuda_memory_binary.mutable_base()->client_addr(),
        /* server_addr */ cuda_memory_binary.mutable_base()->server_addr(),
        /* size */ cuda_memory_binary.mutable_base()->size(),
        /* parent_handles_waitlist */ parent_handles_waitlist,
        /* state_size */ cuda_memory_binary.mutable_base()->state_size()
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to restore mocked resource in handle manager: client_addr(%p)",
            cuda_memory_binary.mutable_base()->client_addr()
        );
        goto exit;
    }
    POS_CHECK_POINTER(*handle);

exit:
    return retval;
}


#if POS_CONF_RUNTIME_EnableTrace

void POSHandleManager_CUDA_Memory::print_metrics() {
    static std::unordered_map<metrics_ticker_type_t, std::string> ticker_names = {
        { RESTORE_reload_state, "Restore State" }
    };
    POS_ASSERT(pos_resource_map.count(this->_rid) > 0);
    POS_LOG(
        "[HandleManager Metrics] %s:\n%s",
        pos_resource_map[this->_rid].c_str(),
        this->metric_tickers.str(ticker_names).c_str()
    );
}

#endif
