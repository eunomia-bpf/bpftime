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

#include "pos/include/command.h"
#include "pos/include/log.h"
#include "pos/include/workspace.h"
#include "pos/cuda_impl/workspace.h"


/*!
 *  \brief  create new workspace for CUDA platform
 *  \return pointer to the created CUDA workspace
 */
static POSWorkspace_CUDA* pos_create_workspace_cuda(){
    pos_retval_t retval = POS_SUCCESS;
    POSWorkspace_CUDA *pos_cuda_ws = nullptr;

    POS_CHECK_POINTER(pos_cuda_ws = new POSWorkspace_CUDA());
    if(unlikely(POS_SUCCESS != (retval = pos_cuda_ws->init()))){
        POS_WARN("failed to initialize PhOS CUDA Workspace: retval(%u)", retval);
        goto exit;
    }

exit:
    if(unlikely(retval != POS_SUCCESS)){
        if(pos_cuda_ws != nullptr){ delete pos_cuda_ws; }
        pos_cuda_ws = nullptr;
    }
    return pos_cuda_ws;
}


/*!
 *  \brief  destory workspace of CUDA platform
 *  \param  pos_cuda_ws pointer to the CUDA workspace to be destoried
 *  \return 0 for successfully destory
 *          1 for failed
 */
static int pos_destory_workspace_cuda(POSWorkspace_CUDA* pos_cuda_ws){
    int retval = 0;
    pos_retval_t pos_retval = POS_SUCCESS;

    POS_CHECK_POINTER(pos_cuda_ws);

    if(unlikely(POS_SUCCESS != (pos_retval = pos_cuda_ws->deinit()))){
        POS_WARN("failed to deinitialize PhOS CUDA Workspace: retval(%u)", pos_retval);
        retval = 1;
        goto exit;
    }
    delete pos_cuda_ws;

exit:
    return retval;
}
