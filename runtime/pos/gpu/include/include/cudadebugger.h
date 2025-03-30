/*
 * Copyright 2007-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


/*--------------------------------- Includes --------------------------------*/

#ifndef CUDADEBUGGER_H
#define CUDADEBUGGER_H

#include <stdlib.h>
#include "cuda_stdint.h"

#if defined(__STDC__)
#include <inttypes.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !defined(_WIN64)
/* Windows 32-bit */
#define PRIxPTR "I32x"
#endif

#if defined(_WIN64)
/* Windows 64-bit */
#define PRIxPTR "I64x"
#endif

#if defined(_WIN32)
/* Windows 32- and 64-bit */
#define PRIx64  "I64x"
#define PRId64  "I64d"
#ifndef __cplusplus
typedef unsigned char bool;
#endif
#undef false
#undef true
#define false 0
#define true  1
#endif

/* OS-agnostic _CUDBG_INLINE */
#if defined(_WIN32)
#define _CUDBG_INLINE __inline
#else
#define _CUDBG_INLINE inline
#endif


/*--------------------------------- API Version ------------------------------*/

#define CUDBG_API_VERSION_MAJOR       7 /* Major release version number */
#define CUDBG_API_VERSION_MINOR       0 /* Minor release version number */
#define CUDBG_API_VERSION_REVISION  122 /* Revision (build) number */

/*---------------------------------- Constants -------------------------------*/

#define CUDBG_MAX_DEVICES 32  /* Maximum number of supported devices */
#define CUDBG_MAX_SMS     64  /* Maximum number of SMs per device */
#define CUDBG_MAX_WARPS   64  /* Maximum number of warps per SM */
#define CUDBG_MAX_LANES   32  /* Maximum number of lanes per warp */

/*----------------------- Thread/Block Coordinates Types ---------------------*/

typedef struct { uint32_t x, y; }    CuDim2;   /* DEPRECATED */
typedef struct { uint32_t x, y, z; } CuDim3;   /* 3-dimensional coordinates for threads,... */

/*--------------------- Memory Segments (as used in DWARF) -------------------*/

typedef enum {
    ptxUNSPECIFIEDStorage,
    ptxCodeStorage,
    ptxRegStorage,
    ptxSregStorage,
    ptxConstStorage,
    ptxGlobalStorage,
    ptxLocalStorage,
    ptxParamStorage,
    ptxSharedStorage,
    ptxSurfStorage,
    ptxTexStorage,
    ptxTexSamplerStorage,
    ptxGenericStorage,
    ptxIParamStorage,
    ptxOParamStorage,
    ptxFrameStorage,
    ptxMAXStorage
} ptxStorageKind;

/*--------------------------- Debugger System Calls --------------------------*/

#define CUDBG_IPC_FLAG_NAME                 cudbgIpcFlag
#define CUDBG_RPC_ENABLED                   cudbgRpcEnabled
#define CUDBG_APICLIENT_PID                 cudbgApiClientPid
#define CUDBG_DEBUGGER_INITIALIZED          cudbgDebuggerInitialized
#define CUDBG_APICLIENT_REVISION            cudbgApiClientRevision
#define CUDBG_SESSION_ID                    cudbgSessionId
#define CUDBG_ATTACH_HANDLER_AVAILABLE      cudbgAttachHandlerAvailable
#define CUDBG_DETACH_SUSPENDED_DEVICES_MASK cudbgDetachSuspendedDevicesMask
#define CUDBG_ENABLE_LAUNCH_BLOCKING        cudbgEnableLaunchBlocking
#define CUDBG_ENABLE_INTEGRATED_MEMCHECK    cudbgEnableIntegratedMemcheck
#define CUDBG_ENABLE_PREEMPTION_DEBUGGING   cudbgEnablePreemptionDebugging
#define CUDBG_RESUME_FOR_ATTACH_DETACH      cudbgResumeForAttachDetach

/*---------------- Internal Breakpoint Entries for Error Reporting ------------*/

#define CUDBG_REPORT_DRIVER_API_ERROR                   cudbgReportDriverApiError
#define CUDBG_REPORT_DRIVER_API_ERROR_FLAGS             cudbgReportDriverApiErrorFlags
#define CUDBG_REPORTED_DRIVER_API_ERROR_CODE            cudbgReportedDriverApiErrorCode
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE  cudbgReportedDriverApiErrorFuncNameSize
#define CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR  cudbgReportedDriverApiErrorFuncNameAddr
#define CUDBG_REPORT_DRIVER_INTERNAL_ERROR              cudbgReportDriverInternalError
#define CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE       cudbgReportedDriverInternalErrorCode

/*----------------------------- API Return Types -----------------------------*/

typedef enum {
    CUDBG_SUCCESS                           = 0x0000,  /* Successful execution */
    CUDBG_ERROR_UNKNOWN                     = 0x0001,  /* Error type not listed below */
    CUDBG_ERROR_BUFFER_TOO_SMALL            = 0x0002,  /* Cannot copy all the queried data into the buffer argument */
    CUDBG_ERROR_UNKNOWN_FUNCTION            = 0x0003,  /* Function cannot be found in the CUDA kernel */
    CUDBG_ERROR_INVALID_ARGS                = 0x0004,  /* Wrong use of arguments (NULL pointer, illegal value,...) */
    CUDBG_ERROR_UNINITIALIZED               = 0x0005,  /* Debugger API has not yet been properly initialized */
    CUDBG_ERROR_INVALID_COORDINATES         = 0x0006,  /* Invalid block or thread coordinates were provided */
    CUDBG_ERROR_INVALID_MEMORY_SEGMENT      = 0x0007,  /* Invalid memory segment requested (read/write) */
    CUDBG_ERROR_INVALID_MEMORY_ACCESS       = 0x0008,  /* Requested address (+size) is not within proper segment boundaries */
    CUDBG_ERROR_MEMORY_MAPPING_FAILED       = 0x0009,  /* Memory is not mapped and cannot be mapped */
    CUDBG_ERROR_INTERNAL                    = 0x000a,  /* A debugger internal error occurred */
    CUDBG_ERROR_INVALID_DEVICE              = 0x000b,  /* Specified device cannot be found */
    CUDBG_ERROR_INVALID_SM                  = 0x000c,  /* Specified sm cannot be found */
    CUDBG_ERROR_INVALID_WARP                = 0x000d,  /* Specified warp cannot be found */
    CUDBG_ERROR_INVALID_LANE                = 0x000e,  /* Specified lane cannot be found */
    CUDBG_ERROR_SUSPENDED_DEVICE            = 0x000f,  /* device is suspended */
    CUDBG_ERROR_RUNNING_DEVICE              = 0x0010,  /* device is running and not suspended */
    CUDBG_ERROR_RESERVED_0                  = 0x0011,  /* Reserved error code */
    CUDBG_ERROR_INVALID_ADDRESS             = 0x0012,  /* address is out-of-range */
    CUDBG_ERROR_INCOMPATIBLE_API            = 0x0013,  /* API version does not match */
    CUDBG_ERROR_INITIALIZATION_FAILURE      = 0x0014,  /* The CUDA Driver failed to initialize */
    CUDBG_ERROR_INVALID_GRID                = 0x0015,  /* Specified grid cannot be found */
    CUDBG_ERROR_NO_EVENT_AVAILABLE          = 0x0016,  /* No event left to be processed */
    CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED    = 0x0017,  /* One or more devices have an associated watchdog (eg. X) */
    CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED     = 0x0018,  /* All devices have an associated watchdog (eg. X) */
    CUDBG_ERROR_INVALID_ATTRIBUTE           = 0x0019,  /* Specified attribute does not exist or is incorrect */
    CUDBG_ERROR_ZERO_CALL_DEPTH             = 0x001a,  /* No function calls have been made on the device */
    CUDBG_ERROR_INVALID_CALL_LEVEL          = 0x001b,  /* Specified call level is invalid */
    CUDBG_ERROR_COMMUNICATION_FAILURE       = 0x001c,  /* Communication error between the debugger and the application. */
    CUDBG_ERROR_INVALID_CONTEXT             = 0x001d,  /* Specified context cannot be found */
    CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM   = 0x001e,  /* Requested address was not originally allocated from device memory (most likely visible in system memory) */
    CUDBG_ERROR_MEMORY_UNMAPPING_FAILED     = 0x001f,  /* Memory is not unmapped and cannot be unmapped */
    CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER = 0x0020,  /* The display driver is incompatible with the API */
    CUDBG_ERROR_INVALID_MODULE              = 0x0021,  /* The specified module is not valid */
    CUDBG_ERROR_LANE_NOT_IN_SYSCALL         = 0x0022,  /* The specified lane is not inside a device syscall */
    CUDBG_ERROR_MEMCHECK_NOT_ENABLED        = 0x0023,  /* Memcheck has not been enabled */
    CUDBG_ERROR_INVALID_ENVVAR_ARGS         = 0x0024,  /* Some environment variable's value is invalid */
    CUDBG_ERROR_OS_RESOURCES                = 0x0025,  /* Error while allocating resources from the OS */
    CUDBG_ERROR_FORK_FAILED                 = 0x0026,  /* Error while forking the debugger process */
    CUDBG_ERROR_NO_DEVICE_AVAILABLE         = 0x0027,  /* No CUDA capable device was found */
    CUDBG_ERROR_ATTACH_NOT_POSSIBLE         = 0x0028,  /* Attaching to the CUDA program is not possible */
    CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE    = 0x0029,  /* The resumeWarpsUntilPC() API is not possible, use resumeDevice() or singleStepWarp() instead */
    CUDBG_ERROR_INVALID_WARP_MASK           = 0x002a,  /* Specified warp mask is zero, or contains invalid warps */
    CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS    = 0x002b,  /* Address cannot be resolved to a GPU unambiguously */
    CUDBG_ERROR_RECURSIVE_API_CALL          = 0x002c,  /* Debug API entry point called from within a debug API callback */
} CUDBGResult;

static const char *CUDBGResultNames[45] = {
    "CUDBG_SUCCESS",
    "CUDBG_ERROR_UNKNOWN",
    "CUDBG_ERROR_BUFFER_TOO_SMALL",
    "CUDBG_ERROR_UNKNOWN_FUNCTION",
    "CUDBG_ERROR_INVALID_ARGS",
    "CUDBG_ERROR_UNINITIALIZED",
    "CUDBG_ERROR_INVALID_COORDINATES",
    "CUDBG_ERROR_INVALID_MEMORY_SEGMENT",
    "CUDBG_ERROR_INVALID_MEMORY_ACCESS",
    "CUDBG_ERROR_MEMORY_MAPPING_FAILED",
    "CUDBG_ERROR_INTERNAL",
    "CUDBG_ERROR_INVALID_DEVICE",
    "CUDBG_ERROR_INVALID_SM",
    "CUDBG_ERROR_INVALID_WARP",
    "CUDBG_ERROR_INVALID_LANE",
    "CUDBG_ERROR_SUSPENDED_DEVICE",
    "CUDBG_ERROR_RUNNING_DEVICE",
    "CUDBG_ERROR_RESERVED_0",
    "CUDBG_ERROR_INVALID_ADDRESS",
    "CUDBG_ERROR_INCOMPATIBLE_API",
    "CUDBG_ERROR_INITIALIZATION_FAILURE",
    "CUDBG_ERROR_INVALID_GRID",
    "CUDBG_ERROR_NO_EVENT_AVAILABLE",
    "CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED",
    "CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED",
    "CUDBG_ERROR_INVALID_ATTRIBUTE",
    "CUDBG_ERROR_ZERO_CALL_DEPTH",
    "CUDBG_ERROR_INVALID_CALL_LEVEL",
    "CUDBG_ERROR_COMMUNICATION_FAILURE",
    "CUDBG_ERROR_INVALID_CONTEXT",
    "CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM",
    "CUDBG_ERROR_MEMORY_UNMAPPING_FAILED",
    "CUDBG_ERROR_INCOMPATIBLE_DISPLAY_DRIVER",
    "CUDBG_ERROR_INVALID_MODULE",
    "CUDBG_ERROR_LANE_NOT_IN_SYSCALL",
    "CUDBG_ERROR_MEMCHECK_NOT_ENABLED",
    "CUDBG_ERROR_INVALID_ENVVAR_ARGS",
    "CUDBG_ERROR_OS_RESOURCES",
    "CUDBG_ERROR_FORK_FAILED",
    "CUDBG_ERROR_NO_DEVICE_AVAILABLE",
    "CUDBG_ERROR_ATTACH_NOT_POSSIBLE",
    "CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE",
    "CUDBG_ERROR_INVALID_WARP_MASK",
    "CUDBG_ERROR_AMBIGUOUS_MEMORY_ADDRESS",
    "CUDBG_ERROR_RECURSIVE_API_CALL"
};

static _CUDBG_INLINE const char *cudbgGetErrorString (CUDBGResult error)
{
    if (((unsigned)error)*sizeof(char *) >= sizeof(CUDBGResultNames))
        return "*UNDEFINED*";
    return CUDBGResultNames[(unsigned)error];
}


/*------------------------- API Error Reporting Flags -------------------------*/
typedef enum {
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_NONE = 0x0000, /* Default is that there is no flag */
    CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_SUPPRESS_NOT_READY = ( 1U << 0 ), /* When set, cudaErrorNotReady/cuErrorNotReady will not be reported */
} CUDBGReportDriverApiErrorFlags;

/*------------------------------ Grid Attributes -----------------------------*/

typedef enum {
    CUDBG_ATTR_GRID_LAUNCH_BLOCKING    = 0x000,   /* Whether the grid launch is blocking or not. */
    CUDBG_ATTR_GRID_TID                = 0x001,   /* Id of the host thread that launched the grid. */
} CUDBGAttribute;

typedef struct {
    CUDBGAttribute attribute;
    uint64_t       value;
} CUDBGAttributeValuePair;

typedef enum {
    CUDBG_GRID_STATUS_INVALID,          /* An invalid grid ID was passed, or an error occurred during status lookup */
    CUDBG_GRID_STATUS_PENDING,          /* The grid was launched but is not running on the HW yet */
    CUDBG_GRID_STATUS_ACTIVE,           /* The grid is currently running on the HW */
    CUDBG_GRID_STATUS_SLEEPING,         /* The grid is on the device, doing a join */
    CUDBG_GRID_STATUS_TERMINATED,       /* The grid has finished executing */
    CUDBG_GRID_STATUS_UNDETERMINED,     /* The grid is either PENDING or TERMINATED */
} CUDBGGridStatus;

/*------------------------------- Kernel Types -------------------------------*/

typedef enum {
    CUDBG_KNL_TYPE_UNKNOWN             = 0x000,   /* Any type not listed below. */
    CUDBG_KNL_TYPE_SYSTEM              = 0x001,   /* System kernel, such as MemCpy. */
    CUDBG_KNL_TYPE_APPLICATION         = 0x002,   /* Application kernel, user-defined or libraries. */
} CUDBGKernelType;

/*--------------------------- Elf Image Properties ---------------------------*/

typedef enum {
    CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM  = 0x001,   /* ELF image contains system kernels. */
} CUDBGElfImageProperties;

/*-------------------------- Physical Register Types -------------------------*/

typedef enum {
    REG_CLASS_INVALID                  = 0x000,   /* invalid register */
    REG_CLASS_REG_CC                   = 0x001,   /* Condition register */
    REG_CLASS_REG_PRED                 = 0x002,   /* Predicate register */
    REG_CLASS_REG_ADDR                 = 0x003,   /* Address register */
    REG_CLASS_REG_HALF                 = 0x004,   /* 16-bit register (Currently unused) */
    REG_CLASS_REG_FULL                 = 0x005,   /* 32-bit register */
    REG_CLASS_MEM_LOCAL                = 0x006,   /* register spilled in memory */
    REG_CLASS_LMEM_REG_OFFSET          = 0x007,   /* register at stack offset (ABI only) */
} CUDBGRegClass;

/*---------------------------- Application Events ----------------------------*/

typedef enum {
    CUDBG_EVENT_INVALID                = 0x000,   /* Invalid event */
    CUDBG_EVENT_ELF_IMAGE_LOADED       = 0x001,   /* ELF image for CUDA kernel(s) is ready */
    CUDBG_EVENT_KERNEL_READY           = 0x002,   /* A CUDA kernel is ready to be launched */
    CUDBG_EVENT_KERNEL_FINISHED        = 0x003,   /* A CUDA kernel has terminated */
    CUDBG_EVENT_INTERNAL_ERROR         = 0x004,   /* Unexpected error. The API may be unstable. */
    CUDBG_EVENT_CTX_PUSH               = 0x005,   /* A CUDA context has been pushed. */
    CUDBG_EVENT_CTX_POP                = 0x006,   /* A CUDA context has been popped. */
    CUDBG_EVENT_CTX_CREATE             = 0x007,   /* A CUDA context has been created and pushed. */
    CUDBG_EVENT_CTX_DESTROY            = 0x008,   /* A CUDA context has been, popped if pushed, then destroyed. */
    CUDBG_EVENT_TIMEOUT                = 0x009,   /* Nothing happened for a while. This is heartbeat event. */
    CUDBG_EVENT_ATTACH_COMPLETE        = 0x00a,   /* Attach complete. */
    CUDBG_EVENT_DETACH_COMPLETE        = 0x00b,   /* Detach complete. */
    CUDBG_EVENT_ELF_IMAGE_UNLOADED     = 0x00c,   /* ELF image for CUDA kernels(s) no longer available */
} CUDBGEventKind;

/*------------------------------- Kernel Origin ------------------------------*/

typedef enum {
    CUDBG_KNL_ORIGIN_CPU               = 0x000,   /* The kernel was launched from the CPU. */
    CUDBG_KNL_ORIGIN_GPU               = 0x001,   /* The kernel was launched from the GPU. */
} CUDBGKernelOrigin;

/*------------------------ Kernel Launch Notify Mode --------------------------*/

typedef enum {
    CUDBG_KNL_LAUNCH_NOTIFY_EVENT      = 0x000,   /* The kernel notifications generate events */
    CUDBG_KNL_LAUNCH_NOTIFY_DEFER      = 0x001,   /* The kernel notifications are deferred */
} CUDBGKernelLaunchNotifyMode;

/*---------------------- Application Event Queue Type ------------------------*/

typedef enum {
    CUDBG_EVENT_QUEUE_TYPE_SYNC      = 0,   /* Synchronous event queue */
    CUDBG_EVENT_QUEUE_TYPE_ASYNC     = 1,   /* Asynchronous event queue */
} CUDBGEventQueueType;

/*------------------------------ Elf Image Type ------------------------------*/

typedef enum {
    CUDBG_ELF_IMAGE_TYPE_NONRELOCATED      = 0,   /* Non-relocated ELF image */
    CUDBG_ELF_IMAGE_TYPE_RELOCATED         = 1,   /* Relocated ELF image */
} CUDBGElfImageType;

/*------------------------------ Code Address --------------------------------*/

typedef enum {
    CUDBG_ADJ_PREVIOUS_ADDRESS         = 0x000,   /* Get the adjusted previous code address. */
    CUDBG_ADJ_CURRENT_ADDRESS          = 0x001,   /* Get the adjusted current code address. */
    CUDBG_ADJ_NEXT_ADDRESS             = 0x002,   /* Get the adjusted next code address. */
} CUDBGAdjAddrAction;

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases30_st {
        struct elfImageLoaded30_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size;
        } elfImageLoaded;
        struct kernelReady30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelReady;
        struct kernelFinished30_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
        } kernelFinished;
    } cases;
} CUDBGEvent30;

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases32_st {
        struct elfImageLoaded32_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
        } elfImageLoaded;
        struct kernelReady32_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelReady;
        struct kernelFinished32_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy32_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
    } cases;
} CUDBGEvent32;

/* Deprecated */
typedef struct {
    CUDBGEventKind kind;
    union cases42_st {
        struct elfImageLoaded42_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady42_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
        } kernelReady;
        struct kernelFinished42_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy42_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
    } cases;
} CUDBGEvent42;

typedef struct {
    CUDBGEventKind kind;
    union cases50_st {
        struct elfImageLoaded50_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady50_st{
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
        } kernelReady;
        struct kernelFinished50_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
        } kernelFinished;
        struct contextPush50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy50_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError50_st {
            CUDBGResult errorType;
        } internalError;
    } cases;
} CUDBGEvent50;

typedef struct {
    CUDBGEventKind kind;
    union cases55_st {
        struct elfImageLoaded55_st {
            char     *relocatedElfImage;
            char     *nonRelocatedElfImage;
            uint32_t  size32;
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
        } elfImageLoaded;
        struct kernelReady55_st{
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
            uint64_t parentGridId;
            uint64_t gridId64;
            CUDBGKernelOrigin origin;
        } kernelReady;
        struct kernelFinished55_st {
            uint32_t dev;
            uint32_t gridId;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            uint64_t gridId64;
        } kernelFinished;
        struct contextPush55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy55_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError55_st {
            CUDBGResult errorType;
        } internalError;
    } cases;
} CUDBGEvent55;

#pragma pack(push,1)
typedef struct {
    CUDBGEventKind kind;
    union cases_st {
        struct elfImageLoaded_st {
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
            uint64_t  handle;
            uint32_t  properties;
        } elfImageLoaded;
        struct elfImageUnloaded_st {
            uint32_t  dev;
            uint64_t  context;
            uint64_t  module;
            uint64_t  size;
            uint64_t  handle;
        } elfImageUnloaded;
        struct kernelReady_st{
            uint32_t dev;
            uint32_t tid;
            uint64_t gridId;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            CuDim3   gridDim;
            CuDim3   blockDim;
            CUDBGKernelType type;
            uint64_t parentGridId;
            CUDBGKernelOrigin origin;
        } kernelReady;
        struct kernelFinished_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
            uint64_t module;
            uint64_t function;
            uint64_t functionEntry;
            uint64_t gridId;
        } kernelFinished;
        struct contextPush_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPush;
        struct contextPop_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextPop;
        struct contextCreate_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextCreate;
        struct contextDestroy_st {
            uint32_t dev;
            uint32_t tid;
            uint64_t context;
        } contextDestroy;
        struct internalError_st {
            CUDBGResult errorType;
        } internalError;
    } cases;
} CUDBGEvent;
#pragma pack(pop)


typedef struct {
    uint32_t tid;
} CUDBGEventCallbackData40;

typedef struct {
    uint32_t tid;
    uint32_t timeout;
} CUDBGEventCallbackData;

#pragma pack(push,1)
typedef struct {
    uint32_t dev;
    uint64_t gridId64;
    uint32_t tid;
    uint64_t context;
    uint64_t module;
    uint64_t function;
    uint64_t functionEntry;
    CuDim3   gridDim;
    CuDim3   blockDim;
    CUDBGKernelType type;
    uint64_t parentGridId;
    CUDBGKernelOrigin origin;
} CUDBGGridInfo;
#pragma pack(pop)

typedef void (*CUDBGNotifyNewEventCallback31)(void *data);
typedef void (*CUDBGNotifyNewEventCallback40)(CUDBGEventCallbackData40 *data);
typedef void (*CUDBGNotifyNewEventCallback)(CUDBGEventCallbackData *data);

/*-------------------------------- Exceptions ------------------------------*/

typedef enum {
    CUDBG_EXCEPTION_UNKNOWN = 0xFFFFFFFFU, // Force sizeof(CUDBGException_t)==4
    CUDBG_EXCEPTION_NONE = 0,
    CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS = 1,
    CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW = 2,
    CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW = 3,
    CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION = 4,
    CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS = 5,
    CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS = 6,
    CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE = 7,
    CUDBG_EXCEPTION_WARP_INVALID_PC = 8,
    CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW = 9,
    CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS = 10,
    CUDBG_EXCEPTION_LANE_MISALIGNED_ADDRESS = 11,
    CUDBG_EXCEPTION_WARP_ASSERT = 12,
    CUDBG_EXCEPTION_LANE_SYSCALL_ERROR = 13,
    CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS = 14,
} CUDBGException_t;

/*------------------------------ Warp State --------------------------------*/
#pragma pack(push,1)
typedef struct {
    uint64_t virtualPC;
    CuDim3 threadIdx;
    CUDBGException_t exception;
} CUDBGLaneState;

typedef struct {
    uint64_t gridId;
    uint64_t errorPC;
    CuDim3 blockIdx;
    uint32_t validLanes;
    uint32_t activeLanes;
    uint32_t errorPCValid;
    CUDBGLaneState lane[32];
} CUDBGWarpState;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct {
    uint64_t startAddress;
    uint64_t size;
} CUDBGMemoryInfo;
#pragma pack(pop)

/*--------------------------------- Exports --------------------------------*/

typedef const struct CUDBGAPI_st *CUDBGAPI;

CUDBGResult cudbgGetAPI(uint32_t major, uint32_t minor, uint32_t rev, CUDBGAPI *api);
CUDBGResult cudbgGetAPIVersion(uint32_t *major, uint32_t *minor, uint32_t *rev);
CUDBGResult cudbgMain(int apiClientPid, uint32_t apiClientRevision, int sessionId, int attachState,
                      int attachEventInitialized, int writeFd, int detachFd, int attachStubInUse,
                      int enablePreemptionDebugging);
void cudbgApiInit(uint32_t arg);
void cudbgApiAttach(void);
void cudbgApiDetach(void);
void CUDBG_REPORT_DRIVER_API_ERROR(void);
void CUDBG_REPORT_DRIVER_INTERNAL_ERROR(void);

extern uint32_t CUDBG_IPC_FLAG_NAME;
extern uint32_t CUDBG_RPC_ENABLED;
extern uint32_t CUDBG_APICLIENT_PID;
extern uint32_t CUDBG_I_AM_DEBUGGER;
extern uint32_t CUDBG_DEBUGGER_INITIALIZED;
extern uint32_t CUDBG_APICLIENT_REVISION;
extern uint32_t CUDBG_SESSION_ID;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_CODE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE;
extern uint64_t CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR;
extern uint64_t CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE;
extern uint32_t CUDBG_ATTACH_HANDLER_AVAILABLE;
extern uint32_t CUDBG_DETACH_SUSPENDED_DEVICES_MASK;
extern uint32_t CUDBG_ENABLE_LAUNCH_BLOCKING;
extern uint32_t CUDBG_ENABLE_INTEGRATED_MEMCHECK;
extern uint32_t CUDBG_ENABLE_PREEMPTION_DEBUGGING;
extern uint32_t CUDBG_RESUME_FOR_ATTACH_DETACH;
extern uint32_t CUDBG_REPORT_DRIVER_API_ERROR_FLAGS;


struct CUDBGAPI_st {
    /* Initialization */
    CUDBGResult (*initialize)(void);
    CUDBGResult (*finalize)(void);

    /* Device Execution Control */
    CUDBGResult (*suspendDevice)(uint32_t dev);
    CUDBGResult (*resumeDevice)(uint32_t dev);
    CUDBGResult (*singleStepWarp40)(uint32_t dev, uint32_t sm, uint32_t wp);

    /* Breakpoints */
    CUDBGResult (*setBreakpoint31)(uint64_t addr);
    CUDBGResult (*unsetBreakpoint31)(uint64_t addr);

    /* Device State Inspection */
    CUDBGResult (*readGridId50)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *gridId);
    CUDBGResult (*readBlockIdx32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *blockIdx);
    CUDBGResult (*readThreadIdx)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
    CUDBGResult (*readBrokenWarps)(uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask);
    CUDBGResult (*readValidWarps)(uint32_t dev, uint32_t sm, uint64_t *validWarpsMask);
    CUDBGResult (*readValidLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *validLanesMask);
    CUDBGResult (*readActiveLanes)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *activeLanesMask);
    CUDBGResult (*readCodeMemory)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readConstMemory)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readGlobalMemory31)(uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readParamMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readSharedMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readLocalMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*readRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
    CUDBGResult (*readPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
    CUDBGResult (*readVirtualPC)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
    CUDBGResult (*readLaneStatus)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, bool *error);

    /* Device State Alteration */
    CUDBGResult (*writeGlobalMemory31)(uint32_t dev, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*writeParamMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*writeSharedMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*writeLocalMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*writeRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);

    /* Grid Properties */
    CUDBGResult (*getGridDim32)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *gridDim);
    CUDBGResult (*getBlockDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockDim);
    CUDBGResult (*getTID)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
    CUDBGResult (*getElfImage32)(uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint32_t *size);

    /* Device Properties */
    CUDBGResult (*getDeviceType)(uint32_t dev, char *buf, uint32_t sz);
    CUDBGResult (*getSmType)(uint32_t dev, char *buf, uint32_t sz);
    CUDBGResult (*getNumDevices)(uint32_t *numDev);
    CUDBGResult (*getNumSMs)(uint32_t dev, uint32_t *numSMs);
    CUDBGResult (*getNumWarps)(uint32_t dev, uint32_t *numWarps);
    CUDBGResult (*getNumLanes)(uint32_t dev, uint32_t *numLanes);
    CUDBGResult (*getNumRegisters)(uint32_t dev, uint32_t *numRegs);

    /* DWARF-related routines */
    CUDBGResult (*getPhysicalRegister30)(uint64_t pc, char *reg, uint32_t *buf, uint32_t sz, uint32_t *numPhysRegs, CUDBGRegClass *regClass);
    CUDBGResult (*disassemble)(uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t sz);
    CUDBGResult (*isDeviceCodeAddress55)(uintptr_t addr, bool *isDeviceAddress);
    CUDBGResult (*lookupDeviceCodeSymbol)(char *symName, bool *symFound, uintptr_t *symAddr);

    /* Events */
    CUDBGResult (*setNotifyNewEventCallback31)(CUDBGNotifyNewEventCallback31 callback, void *data);
    CUDBGResult (*getNextEvent30)(CUDBGEvent30 *event);
    CUDBGResult (*acknowledgeEvent30)(CUDBGEvent30 *event);

    /* 3.1 Extensions */
    CUDBGResult (*getGridAttribute)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttribute attr, uint64_t *value);
    CUDBGResult (*getGridAttributes)(uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttributeValuePair *pairs, uint32_t numPairs);
    CUDBGResult (*getPhysicalRegister40)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t pc, char *reg, uint32_t *buf, uint32_t sz, uint32_t *numPhysRegs, CUDBGRegClass *regClass);
    CUDBGResult (*readLaneException)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
    CUDBGResult (*getNextEvent32)(CUDBGEvent32 *event);
    CUDBGResult (*acknowledgeEvents42)(void);

    /* 3.1 - ABI */
    CUDBGResult (*readCallDepth32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *depth);
    CUDBGResult (*readReturnAddress32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra);
    CUDBGResult (*readVirtualReturnAddress32)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra);

    /* 3.2 Extensions */
    CUDBGResult (*readGlobalMemory55)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*writeGlobalMemory55)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*readPinnedMemory)(uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*writePinnedMemory)(uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*setBreakpoint)(uint32_t dev, uint64_t addr);
    CUDBGResult (*unsetBreakpoint)(uint32_t dev, uint64_t addr);
    CUDBGResult (*setNotifyNewEventCallback40)(CUDBGNotifyNewEventCallback40 callback);

    /* 4.0 Extensions */
    CUDBGResult (*getNextEvent42)(CUDBGEvent42 *event);
    CUDBGResult (*readTextureMemory)(uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t id, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);
    CUDBGResult (*readBlockIdx)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
    CUDBGResult (*getGridDim)(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *gridDim);
    CUDBGResult (*readCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);
    CUDBGResult (*readReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra);
    CUDBGResult (*readVirtualReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra);
    CUDBGResult (*getElfImage)(uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint64_t *size);

    /* 4.1 Extensions */
    CUDBGResult (*getHostAddrFromDeviceAddr)(uint32_t dev, uint64_t device_addr, uint64_t *host_addr);
    CUDBGResult (*singleStepWarp)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *warpMask);
    CUDBGResult (*setNotifyNewEventCallback)(CUDBGNotifyNewEventCallback callback);
    CUDBGResult (*readSyscallCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);

    /* 4.2 Extensions */
    CUDBGResult (*readTextureMemoryBindless)(uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t texSymtabIndex, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);

    /* 5.0 Extensions */
    CUDBGResult (*clearAttachState)(void);
    CUDBGResult (*getNextSyncEvent50)(CUDBGEvent50 *event);
    CUDBGResult (*memcheckReadErrorAddress)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *address, ptxStorageKind *storage);
    CUDBGResult (*acknowledgeSyncEvents)(void);
    CUDBGResult (*getNextAsyncEvent50)(CUDBGEvent50 *event);
    CUDBGResult (*requestCleanupOnDetach55)(void);
    CUDBGResult (*initializeAttachStub)(void);
    CUDBGResult (*getGridStatus50)(uint32_t dev, uint32_t gridId, CUDBGGridStatus *status);

    /* 5.5 Extensions */
    CUDBGResult (*getNextSyncEvent55)(CUDBGEvent55 *event);
    CUDBGResult (*getNextAsyncEvent55)(CUDBGEvent55 *event);
    CUDBGResult (*getGridInfo)(uint32_t dev, uint64_t gridId64, CUDBGGridInfo *gridInfo);
    CUDBGResult (*readGridId)(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId64);
    CUDBGResult (*getGridStatus)(uint32_t dev, uint64_t gridId64, CUDBGGridStatus *status);
    CUDBGResult (*setKernelLaunchNotificationMode) (CUDBGKernelLaunchNotifyMode mode);
    CUDBGResult (*getDevicePCIBusInfo) (uint32_t devId, uint32_t *pciBusId, uint32_t *pciDevId);
    CUDBGResult (*readDeviceExceptionState) (uint32_t devId, uint64_t *exceptionSMMask);

   /* 6.0 Extensions */
    CUDBGResult (*getAdjustedCodeAddress)(uint32_t devId, uint64_t address, uint64_t *adjustedAddress, CUDBGAdjAddrAction adjAction);
    CUDBGResult (*readErrorPC)(uint32_t devId, uint32_t sm, uint32_t wp, uint64_t *errorPC, bool *errorPCValid);
    CUDBGResult (*getNextEvent)(CUDBGEventQueueType type, CUDBGEvent  *event);
    CUDBGResult (*getElfImageByHandle)(uint32_t devId, uint64_t handle, CUDBGElfImageType type, void *elfImage, uint64_t size);
    CUDBGResult (*resumeWarpsUntilPC)(uint32_t devId, uint32_t sm, uint64_t warpMask, uint64_t virtPC);
    CUDBGResult (*readWarpState)(uint32_t devId, uint32_t sm, uint32_t wp, CUDBGWarpState *state);
    CUDBGResult (*readRegisterRange)(uint32_t devId, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t index, uint32_t registers_size, uint32_t *registers);
    CUDBGResult (*readGenericMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*writeGenericMemory)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*readGlobalMemory)(uint64_t addr, void *buf, uint32_t sz);
    CUDBGResult (*writeGlobalMemory)(uint64_t addr, const void *buf, uint32_t sz);
    CUDBGResult (*getManagedMemoryRegionInfo)(uint64_t startAddress, CUDBGMemoryInfo *memoryInfo, uint32_t memoryInfo_size, uint32_t *numEntries);
    CUDBGResult (*isDeviceCodeAddress)(uintptr_t addr, bool *isDeviceAddress);
    CUDBGResult (*requestCleanupOnDetach)(uint32_t appResumeFlag);

   /* 6.5 Extensions */
    CUDBGResult (*readPredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
    CUDBGResult (*writePredicates)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
    CUDBGResult (*getNumPredicates)(uint32_t dev, uint32_t *numPredicates);
    CUDBGResult (*readCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
    CUDBGResult (*writeCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);

    CUDBGResult (*getDeviceName)(uint32_t dev, char *buf, uint32_t sz);
};

#ifdef __cplusplus
}
#endif






#endif
