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
#include <cstdio>
#include <array>
#include <string>
#include <thread>
#include <future>

#include "pos/include/common.h"
#include "pos/include/log.h"

class POSUtil_Command_Caller {
 public:
    /*!
     *  \brief  execute a specified command and obtain its result (synchronously)
     *  \param  cmd             the command to execute
     *  \param  result          the result of the executed command
     *  \param  ignore_error    whether to ignore command execution error (exit_code != 0)
     *  \param  print_stdout    dynamically printing stdout
     *  \param  print_stderr    dynamically printing stderr
     *  \todo   this function should support timeout option
     *  \return POS_SUCCESS once the command is successfully executed
     *          POS_FAILED if failed
     */
    static inline pos_retval_t exec_sync(
        std::string& cmd, std::string& result, 
        bool ignore_error = false, bool print_stdout = false, bool print_stderr = false
    ){
        pos_retval_t retval = POS_SUCCESS;
        std::array<char, 8192> buffer;
        int exit_code = -1;
        FILE *pipe;

        if(print_stderr){
            print_stdout = true;
            cmd = cmd + std::string(" 2>&1");
        }

        pipe = popen(cmd.c_str(), "r");
        if (unlikely(pipe == nullptr)) {
            POS_WARN("failed to open pipe for executing command %s", cmd.c_str());
            retval = POS_FAILED;
            goto exit;
        }

        result.clear();
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
            if(print_stdout){ std::cout << buffer.data(); }
        }

        // remove \n and \r
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
            result.pop_back();
        }

        exit_code = WEXITSTATUS(pclose(pipe));
        if(unlikely(exit_code != 0) && ignore_error == false){
            POS_WARN("failed execution of command %s: exit_code(%d)", cmd.c_str(), exit_code);
            retval = POS_FAILED;
            goto exit;
        }

    exit:
        return retval;
    }


    /*!
     *  \brief  execute a specified command and obtain its result (asynchronously)
     *  \param  cmd             the command to execute
     *  \param  async_thread    thread handle of the async command execution
     *  \param  thread_promise  return value of the async thread
     *  \param  result          the result of the executed command
     *  \param  ignore_error    whether to ignore command execution error (exit_code != 0)
     *  \param  print_stdout    dynamically printing stdout
     *  \param  print_stderr    dynamically printing stderr
     *  \todo   this function should support timeout option
     *  \return POS_SUCCESS once the command is successfully executed
     *          POS_FAILED if failed
     */
    static inline pos_retval_t exec_async(
        std::string& cmd, std::thread& async_thread, std::promise<pos_retval_t>& thread_promise,
        std::string& result, bool ignore_error = false, bool print_stdout = false, bool print_stderr = false
    ){
        pos_retval_t retval = POS_SUCCESS;

        if(print_stderr){
            print_stdout = true;
            cmd = cmd + std::string(" 2>&1");
        }

        auto __exec_async = [&](){
            std::array<char, 8192> buffer;
            int exit_code = -1;
            FILE *pipe;

            pipe = popen(cmd.c_str(), "r");
            if (unlikely(pipe == nullptr)) {
                POS_WARN("failed to open pipe for executing command %s", cmd.c_str());
                retval = POS_FAILED;
                goto async_exit;
            }

            result.clear();
            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                result += buffer.data();
                if(print_stdout){ std::cout << buffer.data(); }
            }

            // remove \n and \r
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }

            exit_code = WEXITSTATUS(pclose(pipe));
            if(unlikely(exit_code != 0) && ignore_error == false){
                POS_WARN("failed execution of command %s: exit_code(%d)", cmd.c_str(), exit_code);
                thread_promise.set_value(POS_FAILED);
            } else {
                thread_promise.set_value(POS_SUCCESS);
            }

        async_exit:
            ;
        };

        async_thread = std::thread(__exec_async);
        async_thread.detach();  

    exit:
        return retval;
    }
};
