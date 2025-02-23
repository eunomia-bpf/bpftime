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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>

#include <sys/time.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <infiniband/verbs.h>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/include/oob.h"
#include "pos/include/utils/timer.h"


#define POS_TRANSPORT_RDMA_MAX_WQE_PER_WQ    128
#define POS_TRANSPORT_RDMA_CQ_SIZE           128
#define POS_TRANSPORT_RDMA_MAX_SGE_PER_WQE   16

/*!
 * \brief   transport end-point
 */
template<bool is_server>
class POSTransport {
 public:
  POSTransport(){}
  ~POSTransport(){}

  /*!
   * \brief   [control-plane] listen to a TCP socket before starting connection,
   *          this function would be invoked on the server-side
   * \return  POS_SUCCESS for succesfully connected
   */
  virtual pos_retval_t handshake(){}

 private:

};

/*!
 * \brief   represent a RDMA-based transport end-point
 */
template<bool is_server>
class POSTransport_RDMA : public POSTransport<is_server> {
 public:
   /*!
    * \brief   constructor of RDMA transport end-point
    * \param   dev_name       name of the IB device to be used
    */
   POSTransport_RDMA(std::string dev_name){ 
      pos_retval_t tmp_retval;

        // make sure ib device exist
        // TODO: temp comment this out
        // POS_ASSERT(POSTransport_RDMA::has_ib_device());
        if(POSTransport_RDMA::has_ib_device() == false){
            goto exit;
        }

        // open and init IB device
        tmp_retval = __open_and_init_ib_device(dev_name);
        if(unlikely(POS_SUCCESS != tmp_retval)){
            goto exit;
        }

        // create Reliable & Connect-oriented (RC) QP and corresponding PD and CQ
        tmp_retval = this->__create_qctx(IBV_QPT_RC);
        if(unlikely(POS_SUCCESS != tmp_retval)){
            goto exit;
        }

   exit:
      ;
   }
   ~POSTransport_RDMA() = default;

   /*!
    * \brief   [control-plane] listen to a TCP socket before starting connection,
    *          this function would be invoked on the server-side
    * \return  POS_SUCCESS for succesfully connected
    */
   pos_retval_t handshake() override {
      pos_retval_t retval = POS_SUCCESS;
      
      if constexpr (is_server == true) {
         
      } else {

      }

   exit:
      return retval;
   }

   /*!
    * \brief   query whether this host contains IB device
    */
   static inline bool has_ib_device(){
      int num_devices;
      ibv_get_device_list(&num_devices);
      return num_devices > 0;
   }

 private:
   /*!
    * \brief   [control-plane] open and initialize specific IB device
    * \param   dev_name       name of the IB device to be used
    * \param   local_ib_port  local IB port to be used
    * \return  POS_SUCCESS for successfully opened;
    *          others for any failure
    */
   pos_retval_t __open_and_init_ib_device(std::string& dev_name){
      pos_retval_t retval = POS_SUCCESS;
      struct ibv_device **dev_list = nullptr;
      struct ibv_qp_init_attr qp_init_attr;
      int i, rc, num_devices;
      char *tmp_dev_name;

      // obtain IB device list
      dev_list = ibv_get_device_list(&num_devices);
      if(unlikely(dev_list == nullptr)){
         POS_WARN_C("failed to obtain IB device list");
         retval = POS_FAILED_NOT_EXIST;
         goto exit;
      }
      if(unlikely(num_devices == 0)){
         POS_WARN_C("no IB device detected");
         retval = POS_FAILED_NOT_EXIST;
         goto exit;
      }
      POS_DEBUG_C("found %d of IB devices", num_devices);

      // decide the used device
      for(i=0; i<num_devices; i++){
         tmp_dev_name = strdup(ibv_get_device_name(dev_list[i]));
         if (!strcmp(tmp_dev_name, dev_name.c_str())){
            this->_ib_dev = dev_list[i];
            break;
         }
      }
      if(dev_name.size() > 0 && this->_ib_dev == nullptr){
         POS_WARN_C("no IB device named %s detected", dev_name.c_str());
         retval = POS_FAILED_NOT_EXIST;
         goto exit;
      }
      if(unlikely(this->_ib_dev == nullptr)){
         this->_ib_dev = dev_list[0];
         POS_DEBUG_C(
            "no IB device specified, use first device by default: dev_name(%s)",
            strdup(ibv_get_device_name(this->_ib_dev))
         );
      }
      POS_CHECK_POINTER(this->_ib_dev);

      // obtain the handle of the IB device
      this->_ib_ctx = ibv_open_device(this->_ib_dev);
      if(unlikely(this->_ib_ctx == nullptr)){
         POS_WARN_C(
            "failed to open IB device handle: device_name(%s)",
            ibv_get_device_name(this->_ib_dev)
         );
         retval = POS_FAILED;
         goto exit;
      }

      // query port properties on the opened device
      if (unlikely(
         0 != ibv_query_port(this->_ib_ctx, 0, &this->_port_attr)
      )){
         POS_WARN_C(
            "failed to ibv_query_port on first port for device %s", strdup(ibv_get_device_name(this->_ib_dev))
         );
         retval = POS_FAILED;
         goto exit;
      }

   exit:
      if(dev_list){
         ibv_free_device_list(dev_list);
      }

      if(unlikely(retval != POS_SUCCESS)){
         if(this->_ib_ctx){
            ibv_close_device(this->_ib_ctx);
            this->_ib_ctx = nullptr;
         }
      }

      return retval;
   }

   /*!
    * \brief   [control-plane] create new queue context (i.e., PD, QP, CQ)
    * \return  POS_SUCCESS for successfully creation
    */
   pos_retval_t __create_qctx(ibv_qp_type qp_type){
      pos_retval_t retval = POS_SUCCESS;
      struct ibv_pd *pd = nullptr;
      struct ibv_qp *qp = nullptr;
      struct ibv_cq *cq = nullptr;
      struct ibv_qp_init_attr qp_init_attr;

      POS_CHECK_POINTER(this->_ib_dev);
      POS_CHECK_POINTER(this->_ib_ctx);

      // allocate completion queue
      cq = ibv_create_cq(this->_ib_ctx, POS_TRANSPORT_RDMA_CQ_SIZE, NULL, NULL, 0);
      if (unlikely(cq == nullptr)){
         POS_WARN_C(
            "failed to create CQ: device(%s), size(%u)",
            strdup(ibv_get_device_name(this->_ib_dev)), POS_TRANSPORT_RDMA_CQ_SIZE
         );
         retval = POS_FAILED;
         goto exit;
	   }

      // allocate protection domain for the QP to be created
      pd = ibv_alloc_pd(this->_ib_ctx);
      if (unlikely(pd == nullptr)){
         POS_WARN_C(
            "failed to allocate protection domain on device %s",
            strdup(ibv_get_device_name(this->_ib_dev))
         );
         retval = POS_FAILED;
         goto exit;
      }

      memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
      qp_init_attr.qp_type = qp_type;
      // if set, each Work Request (WR) submitted to the SQ generates a completion entry
      qp_init_attr.sq_sig_all = 1;
      qp_init_attr.send_cq = cq;
      qp_init_attr.recv_cq = cq;
      // requested max number of outstanding WRs in the SQ/RQ
      qp_init_attr.cap.max_send_wr = POS_TRANSPORT_RDMA_MAX_WQE_PER_WQ;
      qp_init_attr.cap.max_recv_wr = POS_TRANSPORT_RDMA_MAX_WQE_PER_WQ;
      // requested max number of scatter/gather (s/g) elements in a WR in the SQ/RQ
      qp_init_attr.cap.max_send_sge = POS_TRANSPORT_RDMA_MAX_SGE_PER_WQE;
      qp_init_attr.cap.max_recv_sge = POS_TRANSPORT_RDMA_MAX_SGE_PER_WQE;

      qp = ibv_create_qp(pd, &qp_init_attr);
      if (unlikely(qp == nullptr)){
         POS_WARN_C("failed to create qp on IB device %s", strdup(ibv_get_device_name(this->_ib_dev)));
         retval = POS_FAILED;
         goto exit;
      }

      POS_DEBUG_C(
         "create queue context: device(%s), max_send/recv_wr(%u), max_send/recv_sge(%u), cq_size(%u) ",
         strdup(ibv_get_device_name(this->_ib_dev)),
         POS_TRANSPORT_RDMA_MAX_WQE_PER_WQ,
         POS_TRANSPORT_RDMA_MAX_SGE_PER_WQE,
         POS_TRANSPORT_RDMA_CQ_SIZE
      );
      this->_pd = pd;
      this->_qp = qp;
      this->_cq = cq;

   exit:
      if(unlikely(retval != POS_SUCCESS)){
         if(cq != nullptr){
            ibv_destroy_cq(cq);
         }

         if(pd != nullptr){
            ibv_dealloc_pd(pd);
         }

         if(qp != nullptr){
            ibv_destroy_qp(qp);
         }
      }

      return retval;
   }

   // IB device handle
   struct ibv_device *_ib_dev;

   // IB context of current process
	 struct ibv_context *_ib_ctx;

   // IB port attributes
   struct ibv_port_attr _port_attr;
      
   // structures for IB queues
   ibv_pd *_pd;
   ibv_cq *_cq;
   ibv_qp *_qp;

   // OOB server / client
   POSOobServer *_oob_server;
   POSOobClient *_oob_client;
};
