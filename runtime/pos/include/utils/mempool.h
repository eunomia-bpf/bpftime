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
#include <algorithm>

#include <stdint.h>
#include <string.h>

#include "pos/include/common.h"

enum pos_mempool_elt_state_t {
    kPOS_Mempool_Elt_State_Free=0,
    kPOS_Mempool_Elt_State_Occupied
};

/*!
 *  \brief  descriptor for element within the mempool
 */
typedef struct POSMempoolElt {
    void *base_addr;
    uint64_t id;
    uint64_t size;
    pos_mempool_elt_state_t state;
} POSMempoolElt_t;

/*!
 *  \brief  memory pool
 *  \tparam nb_elts     number of element within the mempool
 *  \tparam elt_size    size of each mempool element
 */
template<uint64_t t_nb_elts, uint64_t t_elt_size>
struct POSMempool {
    /*!
     *  \brief  construtor
     */
    POSMempool() : nb_elts(t_nb_elts), elt_size(t_elt_size), nb_free_elts(t_nb_elts) {
        uint64_t i;
        
        memset(_desps, 0, t_nb_elts*sizeof(POSMempoolElt_t));
        memset(_raw_mem, 0, t_elt_size*t_nb_elts*sizeof(uint8_t));

        for(i=0; i<t_nb_elts; i++){ 
            _desps[i].base_addr = (void*)((uint64_t)_raw_mem+i*t_elt_size);
            _desps[i].state = kPOS_Mempool_Elt_State_Free;
            _desps[i].id = i;
            _desps[i].size = 0;
        }
    }

    ~POSMempool() = default;

    /*!
     *  \brief  get the first free element within the mempool
     *  \return nullptr for no free element; non-nullptr for pointer to
     *          the founded element
     */
    inline std::vector<POSMempoolElt_t*> get_free_elts(uint64_t nb_elts){
        uint64_t i;
        std::vector<POSMempoolElt_t*> ret_vec;

        if(unlikely(nb_free_elts == 0)){ return ret_vec; }

        for(i=0; i<t_nb_elts; i++){
            if(_desps[i].state == kPOS_Mempool_Elt_State_Free){
                _desps[i].state = kPOS_Mempool_Elt_State_Occupied;
                nb_free_elts -= 1;
                ret_vec.push_back(&_desps[i]);
                if(ret_vec.size() == nb_elts){ break; }
            }
        }

        return ret_vec;
    }

    /*!
     *  \brief  get the element by specified index
     *  \param  id  the specified element id
     *  \return nullptr for invalid id that exceed range;
     *          non-nullptr for pointer to the founded element
     */
    inline POSMempoolElt_t* get_elt_by_id(uint64_t id){
        if(unlikely(id >= t_nb_elts)){
            return nullptr;
        } else {
            return &(_desps[id]);
        }
    }

    /*!
     *  \brief  get the element by base address
     *  \param  addr    base address of the element to be returned
     *  \return nullptr for invalid base address that exceed range;
     *          non-nullptr for pointer to the founded element
     */
    inline POSMempoolElt_t* get_elt_by_addr(void* addr){
        uint64_t elt_id;
        if((uint64_t)addr < (uint64_t)_raw_mem || (uint64_t)addr > (uint64_t)_raw_mem + sizeof(_raw_mem)){
            return nullptr;
        }
        elt_id = ((uint64_t)addr - (uint64_t)_raw_mem) / t_elt_size;
        return &_desps[elt_id];
    }

    /*!
     *  \brief  return back the element to the mempool
     *  \param  elt the element to be returned
     */
    inline void return_elt(POSMempoolElt_t* elt){
        POS_CHECK_POINTER(elt);
        if(likely(elt->state != kPOS_Mempool_Elt_State_Free)){
            elt->state = kPOS_Mempool_Elt_State_Free;
            memset(elt->base_addr, 0, t_elt_size);
            elt->size = 0;
            nb_free_elts += 1;
        } else {
            POS_WARN_C_DETAIL(
                "try to return a free element, this is a bug: base_addr(%p), size(%lu)",
                elt->base_addr, elt->size
            );
        }
    }

    uint64_t nb_elts;

    uint64_t nb_free_elts;

    uint64_t elt_size;

    // descriptors for mempool elements
    POSMempoolElt_t _desps[t_nb_elts];

    // payloads
    uint8_t _raw_mem[t_nb_elts*t_elt_size];
};

template<uint64_t t_nb_elts, uint64_t t_elt_size>
using POSMempool_t = struct POSMempool<t_nb_elts, t_elt_size>;
