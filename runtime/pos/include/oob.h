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
#include <thread>
#include <map>
#include <set>

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include "pos/include/common.h"
#include "pos/include/log.h"

class POSWorkspace;
class POSAgent;
class POSOobServer;
class POSOobClient;


/*!
 *  \brief  metadata of a out-of-band client
 */
typedef struct POSOobClientMeta {
    // ip address
    in_addr_t ipv4;

    // udp port
    uint16_t port;

    // process id on the host
    __pid_t pid;

    // uuid of the client on the server
    pos_client_uuid_t uuid;

    // id of the session
    uint64_t session_id = 0;
} POSOobClientMeta_t;


/*!
 *  \brief  out-of-band message type id
 */
enum pos_oob_msg_typeid_t {
    kPOS_OOB_Msg_Unknown=0,

    // ========== OOB management ==========
    kPOS_OOB_Msg_Mgnt_OpenSession,
    kPOS_OOB_Msg_Mgnt_CloseSession,

    // ========== agent message ==========
    kPOS_OOB_Msg_Agent_Register_Client,
    kPOS_OOB_Msg_Agent_Unregister_Client,

    // ========== cli message ==========
    /*!
     *  \note   checkpoint / restore
     */
    kPOS_OOB_Msg_CLI_Ckpt_PreDump,
    kPOS_OOB_Msg_CLI_Ckpt_Dump,
    kPOS_OOB_Msg_CLI_Restore,
    /*!
     *  \note   trace
     */
    kPOS_OOB_Msg_CLI_Trace_Resource,
    kPOS_OOB_Msg_CLI_Trace_Performance,
    /*!
     *  \note   migration
     */
    kPOS_OOB_Msg_CLI_Migration_RemotePrepare,
    kPOS_OOB_Msg_CLI_Migration_LocalPrepare,
    kPOS_OOB_Msg_CLI_Migration_Signal,

    // ========== util message ==========
    kPOS_OOB_Msg_Utils_MockAPICall
};


/*!
 *  \brief  residing state of a session
 */
typedef struct POSOobSession {
    // sock fd for this session
    int fd = 0;
    
    // server-side UDP port
    uint16_t server_port;

    // socket address of this session
    struct sockaddr_in sock_addr;

    // whether to force quiting the session
    bool quit_flag = false;

    // handle of the session thread
    std::thread *daemon = nullptr;

    // mark whether this is the main session
    bool main_session = false;
} POSOobSession_t;


/*!
 *  \brief  out-of-band message content
 */
typedef struct POSOobMsg {
    // type of the message
    pos_oob_msg_typeid_t msg_type;

    // meta data of a out-of-band client
    POSOobClientMeta_t client_meta;

    /*!
     *  \brief  pointer to the corresponding session structure
     *  \note   this field is valid only on server-side
     */
    POSOobSession_t *session = nullptr;

    // out-of-band message payload
#define POS_OOB_MSG_MAXLEN 1024
    uint8_t payload[POS_OOB_MSG_MAXLEN];
} POSOobMsg_t;


/*!
 *  \brief  payload of the request to create a new session
 */
typedef struct oob_payload_create_session {
    /* client */    
    /* server */
    uint64_t session_id;
    uint16_t port;
} oob_payload_create_session_t;


/*!
 *  \brief  default endpoint config of OOB server
 */
#define POS_OOB_SERVER_DEFAULT_IP   "0.0.0.0"
#define POS_OOB_SERVER_DEFAULT_PORT 5213
#define POS_OOB_CLIENT_DEFAULT_PORT 12123


/*!
 *  \brief  prototype of the server-side function
 */
using oob_server_function_t = pos_retval_t(*)(int, struct sockaddr_in*, POSOobMsg_t*, POSWorkspace*, POSOobServer*);


/*!
 *  \brief  prototype of the client-side function
 */
using oob_client_function_t = pos_retval_t(*)(int, struct sockaddr_in*, POSOobMsg_t*, POSAgent*, POSOobClient*, void*);


/*!
 *  \brief  macro for sending OOB message between client & server
 */
#define __POS_OOB_SEND()                                                                                                \
{                                                                                                                       \
    if(unlikely(sendto(fd, msg, sizeof(POSOobMsg_t), 0, (struct sockaddr*)remote, sizeof(struct sockaddr_in)) < 0)){    \
        POS_WARN_DETAIL("failed oob sending: %s", strerror(errno));                                                     \
        return POS_FAILED_NETWORK;                                                                                      \
    }                                                                                                                   \
}


/*!
 *  \brief  macro for receiving OOB message between client & server
 */
#define __POS_OOB_RECV()                                                                                                \
{                                                                                                                       \
    socklen_t __socklen__ = sizeof(struct sockaddr_in);                                                                 \
    if(unlikely(recvfrom(fd, msg, sizeof(POSOobMsg_t), 0, (struct sockaddr*)remote, &__socklen__) < 0)){                \
        POS_WARN_DETAIL("failed oob sending: %s", strerror(errno));                                                     \
        return POS_FAILED_NETWORK;                                                                                      \
    }                                                                                                                   \
}


/*!
 *  \brief  macro for declare server-side handlers / client-side functions
 */
namespace oob_functions {
#define POS_OOB_DECLARE_SVR_FUNCTIONS(oob_type) namespace oob_type {                                                    \
    pos_retval_t sv(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSWorkspace* ws, POSOobServer* oob_server);  \
}

#define POS_OOB_DECLARE_CLNT_FUNCTIONS(oob_type) namespace oob_type {                                                                   \
    pos_retval_t clnt(int fd, struct sockaddr_in* remote, POSOobMsg_t* msg, POSAgent* agent, POSOobClient* oob_clnt, void *call_data);  \
}
}; // namespace oob_functions


/*!
 *  \brief  UDP-based out-of-band RPC server
 */
class POSOobServer {
 public:
    /*!
     *  \brief  constructor
     *  \param  ws                  the workspace that include current oob server
     *  \param  callback_handlers   callback handlers of this OOB server
     *  \param  ip_str              ip address to bind
     *  \param  port                udp port to bind
     */
    POSOobServer(
        POSWorkspace* ws,
        std::map<pos_oob_msg_typeid_t, oob_server_function_t> callback_handlers,
        const char *ip_str=POS_OOB_SERVER_DEFAULT_IP,
        uint16_t port=POS_OOB_SERVER_DEFAULT_PORT
    ) : _ws(ws) {
        pos_retval_t retval;
        POSOobSession_t *session;

        POS_CHECK_POINTER(ws);

        // step 1: insert oob callback map
        _callback_map.insert(callback_handlers.begin(), callback_handlers.end());

        // step 2: create main session
        retval = this->create_new_session</* is_main_session */ true>(&session);
        if(unlikely(retval != POS_SUCCESS)){
            POS_ERROR_C_DETAIL("failed to create OOB main session");
        }
    }


    /*!
     *  \brief  deconstructor
     */
    ~POSOobServer(){ shutdown(); }


    /*!
     *  \brief  raise the shutdown signal to stop the daemon
     */
    inline void shutdown(){
        pos_retval_t tmp_retval;
        typename std::map<uint16_t, POSOobSession_t*>::iterator session_map_iter;

        for(session_map_iter = this->_session_map.begin(); session_map_iter != this->_session_map.end(); session_map_iter++) {
            tmp_retval = this->__shutdown_session(session_map_iter->first);
            if(unlikely(tmp_retval != POS_SUCCESS)){
                POS_WARN_C("failed to shutdown session: udp_port(%u), retval(%u)", session_map_iter->first, tmp_retval);
            } else {
                POS_DEBUG_C("shutdown session: udp_port(%u)", session_map_iter->first);
            }
        }

        // remove all sessions from session map
        this->_session_map.clear();
    }


    /*!
     *  \brief  processing daemon of the session thread
     *  \tparam is_main_session     mark whether is the main session
     *  \param  session handle of the session
     */
    template<bool is_main_session>
    void session_daemon(POSOobSession_t *session){
        pos_retval_t retval = POS_SUCCESS;
        int sock_retval;
        struct sockaddr_in remote_addr;
        socklen_t len = sizeof(remote_addr);
        uint8_t recvbuf[sizeof(POSOobMsg)] = {0};
        POSOobMsg *recvmsg;
        typename std::set<uint16_t>::iterator port_set_iter;

        POS_CHECK_POINTER(session);

        while(session->quit_flag == false){
            memset(recvbuf, 0, sizeof(recvbuf));
            sock_retval = recvfrom(session->fd, recvbuf, sizeof(recvbuf), 0, (struct sockaddr*)&remote_addr, &len);
            if(sock_retval < 0){
                if(errno == EAGAIN){
                    continue;
                } else {
                    POS_WARN_C("failed to recv oob message, daemon stop due to socket broken: errno(%d)", errno);
                    break;
                }
            }
            
            recvmsg = (POSOobMsg*)recvbuf;
            POS_DEBUG_C(
                "oob recv info: recvmsg.msg_type(%lu), recvmsg.client(ip: %u, port: %u, pid: %d)",
                recvmsg->msg_type, recvmsg->client_meta.ipv4, recvmsg->client_meta.port, recvmsg->client_meta.pid
            );

            // invoke corresponding callback function
            if(unlikely(_callback_map.count(recvmsg->msg_type)) == 0){
                POS_ERROR_C_DETAIL(
                    "no callback function register for oob msg type %d, this is a bug",
                    recvmsg->msg_type
                )
            }
            retval = (*(_callback_map[recvmsg->msg_type]))(session->fd, &remote_addr, recvmsg, _ws, this);
            if(unlikely(retval != POS_SUCCESS)){
                POS_WARN_C("failed to execute OOB function: retval(%u)", retval);
            }

            // clean closed session
            if constexpr (is_main_session == true) {
                for(port_set_iter = this->_close_session_ports.begin(); port_set_iter != this->_close_session_ports.end(); port_set_iter++){
                    retval = this->__shutdown_session(*port_set_iter);
                    if(unlikely(retval != POS_SUCCESS)){
                        POS_WARN_C("failed to clean closed session: retval(%u), udp_port(%u)", retval, *port_set_iter);
                    }
                    this->_session_map.erase(*port_set_iter);
                }
                this->_close_session_ports.clear();
            }
        }
    }


    /*!
     *  \brief  create a new session
     *  \tparam is_main_session     mark whether is the main session
     *  \param  new_session         pointer to stored the created session
     *  \param  ip_str              specified ip address to bind this session on
     *  \param  port                specified UDP port to bind this session on
     *                              (for main session, it must be POS_OOB_SERVER_DEFAULT_PORT)
     *  \return POS_SUCCESS for succesfully session creation
     */
    template<bool is_main_session>
    pos_retval_t create_new_session(
        POSOobSession_t **new_session,
        const char *ip_str=POS_OOB_SERVER_DEFAULT_IP,
        uint16_t port=POS_OOB_SERVER_DEFAULT_PORT
    ){
        pos_retval_t retval = POS_SUCCESS;
        uint8_t retry_time = 1;
        struct sockaddr_in spec_addr, res_addr;
        uint16_t new_session_port;
        int fd_flag;
        uint32_t tmp_size;

        POS_CHECK_POINTER(new_session);

        *new_session = new POSOobSession_t();
        POS_CHECK_POINTER(*new_session);

        // create new socket
        (*new_session)->fd = socket(AF_INET, SOCK_DGRAM, 0);
        if ((*new_session)->fd < 0) {
            POS_WARN_C("failed to create socket for new session: error(%s)", strerror(errno));
            retval = POS_FAILED;
            goto exit;
        }

        /*!
         *  \brief  try bind the socket to a local address
         *  \note   for the creation of main session, we bind the socket to a specific UDP port; for side session
         *          required by client, we bind the socket to a random UDP port
         */
        if constexpr (is_main_session == true){
            POS_ASSERT(port == POS_OOB_SERVER_DEFAULT_PORT);
            (*new_session)->sock_addr.sin_family = AF_INET;
            (*new_session)->sock_addr.sin_addr.s_addr = inet_addr(ip_str);
            (*new_session)->sock_addr.sin_port = htons(port);
            if(bind((*new_session)->fd, (struct sockaddr*)&((*new_session)->sock_addr), sizeof((*new_session)->sock_addr)) < 0){
                POS_WARN_C(
                    "failed to obtain socket address for the main session: %s", strerror(errno)
                );
                retval = POS_FAILED;
                goto exit;
            }
            (*new_session)->server_port = port;
            POS_DEBUG_C("OOB main session is binded to %s:%u", ip_str, port);
        } else { // if constexpr (is_main_session == false)
            spec_addr.sin_family = AF_INET;
            spec_addr.sin_addr.s_addr = inet_addr(ip_str);
            spec_addr.sin_port = htons(0);
            while(bind((*new_session)->fd, (struct sockaddr*)&spec_addr, sizeof(spec_addr)) != 0){
                if(retry_time == 512){
                    POS_WARN_C("failed to bind socket address for session: error(%s), retry_time(%u)", strerror(errno), retry_time);
                    retval = POS_FAILED_DRAIN;
                    goto exit;
                }
                retry_time += 1;
            }
            tmp_size = sizeof((*new_session)->sock_addr); 
            if(getsockname((*new_session)->fd, (struct sockaddr *)(&((*new_session)->sock_addr)), &tmp_size) < 0){
                POS_WARN_C(
                    "failed to obtain socket address for the new side session: %s", strerror(errno)
                );
                retval = POS_FAILED;
                goto exit;
            }
            (*new_session)->server_port = ntohs((*new_session)->sock_addr.sin_port);
            POS_ASSERT((*new_session)->server_port != POS_OOB_SERVER_DEFAULT_PORT);
            POS_DEBUG("create new side session: udp_port(%u)", (*new_session)->server_port);
        }

        // set socket as non-block
        fd_flag = fcntl((*new_session)->fd, F_GETFL, 0);
        fcntl((*new_session)->fd, F_SETFL, fd_flag|O_NONBLOCK);

        // create handle thread for the session
        (*new_session)->daemon = new std::thread(&POSOobServer::session_daemon<is_main_session>, this, *new_session);
        POS_CHECK_POINTER((*new_session)->daemon);

        // insert the newly created session to the map
        POS_ASSERT(this->_session_map.count((*new_session)->server_port) == 0);
        this->_session_map[(*new_session)->server_port] = (*new_session);

    exit:
        if(unlikely(retval != POS_SUCCESS)){
            // release session resource
            if((*new_session) != nullptr){
                // 1. close socket
                if((*new_session)->fd > 0){
                    close((*new_session)->fd);
                }

                // 2. stop daemon thread for the session
                if((*new_session)->daemon != nullptr){
                    (*new_session)->quit_flag = true;
                    (*new_session)->daemon->join();
                    delete (*new_session)->daemon;
                }

                // 3. delete session context
                delete (*new_session);
            }
        }

        return retval;
    }

    /*!
     *  \brief  mark session as closed
     *  \param  port    UDP port of the session to be closed
     *  \return POS_SUCCESS for successful closure;
     *          POS_FAILED_NOT_EXIST for unexist session
     */
    pos_retval_t mark_session_closed(uint16_t port){
        pos_retval_t retval = POS_SUCCESS;

        if(unlikely(this->_session_map.count(port) == 0)){
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }
        this->_close_session_ports.insert(port);

    exit:
        return retval;
    }

 private:
    /*!
     *  \brief  remove old session with specified UDP port
     *  \note   this function should only be called within the main session
     *  \param  port    specified UDP port
     *  \return POS_SUCCESS for succesfully remove
     */
    pos_retval_t __shutdown_session(uint16_t port){
        pos_retval_t retval = POS_SUCCESS;
        POSOobSession_t *session;

        if(unlikely(this->_session_map.count(port) == 0)){
            POS_WARN_C("failed to remove session, no session with specified UDP port exit: udp_port(%u)", port);
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        session = this->_session_map[port];
        POS_CHECK_POINTER(session);

        // 1. stop daemon thread for the session
        if(session->daemon != nullptr){
            session->quit_flag = true;
            session->daemon->join();
            delete session->daemon;
        }

        // 2. close socket
        if(session->fd > 0){
            close(session->fd);
        }

        // 3. delete session context
        delete session;

    exit:
        return retval;
    }

    // map of callback functions
    std::map<pos_oob_msg_typeid_t, oob_server_function_t> _callback_map;

    // pointer to the server-side workspace
    POSWorkspace *_ws;

    // map of sessions (udp port -> session context)
    std::map<uint16_t, POSOobSession_t*> _session_map;

    // port of session to be closed
    std::set<uint16_t> _close_session_ports;
};


/*!
 *  \brief  UDP-based out-of-band RPC client
 */
class POSOobClient {
 public:
    /*!
     *  \brief  constructor
     *  \param  agent           pointer to the client-side agent
     *  \param  req_functions   request handlers of this OOB client
     *  \param  local_port      expected local port to bind
     *  \param  local_ip        exepected local ip to bind
     *  \param  server_port     destination server port
     *  \param  server_ip       destination server ipv4
     */
    POSOobClient(
        POSAgent *agent,
        std::map<pos_oob_msg_typeid_t, oob_client_function_t> req_functions,
        uint16_t local_port,
        const char* local_ip,
        uint16_t server_port,
        const char* server_ip
    ) : _agent(agent) {
        __init(req_functions, local_port, local_ip, server_port, server_ip);
    }

    /*!
     *  \brief  constructor
     *  \param  req_functions   request handlers of this OOB client
     *  \param  local_port      expected local port to bind
     *  \param  local_ip        exepected local ip to bind
     *  \param  server_port     destination server port
     *  \param  server_ip       destination server ipv4
     */
    POSOobClient(
        std::map<pos_oob_msg_typeid_t, oob_client_function_t> req_functions,
        uint16_t local_port,
        const char* local_ip,
        uint16_t server_port,
        const char* server_ip
    ) : _agent(nullptr) {
        __init(req_functions, local_port, local_ip, server_port, server_ip);
    }

    /*!
     *  \brief  constructor
     *  \param  req_functions   request handlers of this OOB client
     *  \param  local_port      expected local port to bind
     *  \param  local_ip        exepected local ip to bind
     */
    POSOobClient(
        std::map<pos_oob_msg_typeid_t, oob_client_function_t> req_functions,
        uint16_t local_port,
        const char* local_ip
    ) : _agent(nullptr) {
        __init(req_functions, local_port, local_ip);
    }
    
    /*!
     *  \brief  call OOB RPC request procedure according to OOB message type
     *  \param  id          the OOB message type
     *  \param  call_data   calling payload, coule bd null
     *  \return POS_SUCCESS for successfully requesting
     */
    inline pos_retval_t call(pos_oob_msg_typeid_t id, void *call_data){
        if(unlikely(_request_map.count(id) == 0)){
            POS_ERROR_C_DETAIL("no request function for type %d is registered, this is a bug", id);
        }
        return (*(_request_map[id]))(_fd, &_remote_addr, &_msg, _agent, this, call_data);
    }

    /*!
     *  \brief  call OOB RPC request procedure according to OOB message type
     *  \param  id          the OOB message type
     *  \param  server_port destination server port
     *  \param  server_ip   destination server ipv4
     *  \param  call_data   calling payload, coule bd null
     *  \return POS_SUCCESS for successfully requesting
     */
    inline pos_retval_t call(
        pos_oob_msg_typeid_t id,
        uint16_t server_port=POS_OOB_SERVER_DEFAULT_PORT,
        const char* server_ip="127.0.0.1",
        void *call_data=nullptr
    ){
        struct sockaddr_in remote_addr;

        if(unlikely(_request_map.count(id) == 0)){
            POS_ERROR_C_DETAIL("no request function for type %d is registered, this is a bug", id);
        }

        // setup server addr
        remote_addr.sin_family = AF_INET;
        remote_addr.sin_addr.s_addr = inet_addr(server_ip);
        remote_addr.sin_port = htons(server_port);

        return (*(_request_map[id]))(_fd, &remote_addr, &_msg, _agent, this, call_data);
    }

    /*!
     *  \brief  deconstrutor
     */
    ~POSOobClient(){
        if(_fd > 0){ close(_fd); }
    }

    /*!
     *  \brief  set the uuid of the client
     *  \note   this function is invoked during the registeration process 
     *          (i.e., agent_register_client oob type)
     */
    inline void set_uuid(pos_client_uuid_t id){ _msg.client_meta.uuid = id; }

 private:
    /*!
     *  \brief  internal inialization function of oob client
     *  \param  req_functions   request handlers of this OOB client
     *  \param  local_port      local UDP port to bind
     *  \param  local_ip        local IPv4 to bind
     *  \param  server_port     remote UDP port to send UDP datagram
     *  \param  server_ip       remote IPv4 address pf POS server process
     */
    inline void __init(
        std::map<pos_oob_msg_typeid_t, oob_client_function_t> &req_functions,
        uint16_t local_port,
        const char* local_ip="0.0.0.0",
        uint16_t server_port=POS_OOB_SERVER_DEFAULT_PORT,
        const char* server_ip="127.0.0.1"
    ){
        uint8_t retry_time = 1;

        // step 1: insert oob request map
        _request_map.insert(req_functions.begin(), req_functions.end());

        // step 2: obtain the process id
        _msg.client_meta.pid = getpid();

        // step 3: create socket
        _fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_fd < 0) {
            POS_ERROR_C_DETAIL(
                "failed to create _fd for out-of-band UDP client: %s",
                strerror(errno)
            );
        }
        _port = local_port;
        _local_addr.sin_family = AF_INET;
        _local_addr.sin_addr.s_addr = inet_addr(local_ip);
        _local_addr.sin_port = htons(_port);
        while(bind(_fd, (struct sockaddr*)&_local_addr, sizeof(_local_addr)) < 0){
            if(retry_time == 100){
                POS_ERROR_C_DETAIL("failed to bind oob client to local port, too many clients? try increase the threashold");
            }
            POS_WARN_C(
                "failed to bind out-of-band UDP client to \"%s:%u\": %s, try %uth time to switch port to %u",
                local_ip, _port, strerror(errno), retry_time, _port+1
            );
            retry_time += 1;
            _port += 1;
            _local_addr.sin_port = htons(_port);
        }
        POS_DEBUG_C("out-of-band UDP client is binded to %s:%u", local_ip, _port);
        _msg.client_meta.ipv4 = inet_addr(local_ip);
        _msg.client_meta.port = _port;

        // setup server addr
        _remote_addr.sin_family = AF_INET;
        _remote_addr.sin_addr.s_addr = inet_addr(server_ip);
        _remote_addr.sin_port = htons(server_port);
    }

    // UDP socket
    int _fd;

    // local-used port
    uint16_t _port;

    // local and remote address
    struct sockaddr_in _local_addr, _remote_addr;

    // the one-and-only oob message instance
    POSOobMsg_t _msg;

    // pointer to the client-side POS agent
    POSAgent *_agent;

    // map of request functions
    std::map<pos_oob_msg_typeid_t, oob_client_function_t> _request_map;
};
