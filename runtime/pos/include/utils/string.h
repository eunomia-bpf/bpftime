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
#include <sstream>
#include <string>
#include <vector>

#include "pos/include/common.h"
#include "pos/include/log.h"

class POSUtil_String {
 public:
    /*!
     *  \brief      extract substring field
     *  \example    void kernel(void* p1, int p1) -> (void* p1, int p1)
     *  \tparam     reverse         whether to extract in reverse order
     *  \param      left_sign       left part of the field, e.g., '('
     *  \param      right_sign      right part of the field, e.g., ')'
     *  \param      target_str      target string to be processed
     *  \param      result_str      extracted result string
     *  \return     POS_SUCCESS for successfully extracted
     */
    template<bool reverse>
    static pos_retval_t extract_substring_from_field(
        char left_sign, char right_sign, const std::string& target_str, std::string& result_str
    ){
        pos_retval_t retval = POS_SUCCESS;
        int64_t i, nb_skip;
        uint64_t left_sign_pos = std::string::npos, right_sign_pos = std::string::npos;

        nb_skip = 0;
        result_str.clear();

        if constexpr (reverse){
            right_sign_pos = target_str.find_last_of(right_sign, std::string::npos);
            if (right_sign_pos == std::string::npos){
                // no right sign founded when extract in reverse order, this is not a valid input
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
            for(i=right_sign_pos-1; i>=0; i--){
                if(unlikely(target_str[i] == right_sign)){
                    nb_skip += 1;
                }
                if(unlikely(target_str[i] == left_sign)){
                    if(nb_skip == 0){
                        left_sign_pos = i;
                        break;
                    } else {
                        nb_skip--;
                    }
                }
            }
        } else {
            left_sign_pos = target_str.find(left_sign);
            if (left_sign_pos == std::string::npos){
                // no left sign founded when extract in forward order, this is not a valid input
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
            for(i=left_sign_pos+1; i<target_str.length(); i++){
                if(unlikely(target_str[i] == left_sign)){
                    nb_skip += 1;
                }
                if(unlikely(target_str[i] == right_sign)){
                    if(nb_skip == 0){
                        right_sign_pos = i;
                        break;
                    } else {
                        nb_skip--;
                    }
                }
            }
        }

        if(unlikely(
            nb_skip > 0 || left_sign_pos == std::string::npos || right_sign_pos == std::string::npos 
        )){
            retval = POS_FAILED_INVALID_INPUT;
        } else {
            result_str = target_str.substr(left_sign_pos, right_sign_pos-left_sign_pos+1);
        }

    exit:
        return retval;
    }

    /*!
     *  \brief  splict a string based on given delimiter
     *  \param  str         given string to be splited
     *  \param  delimiter   the delimiter
     *  \return sub strings
     */
    static std::vector<std::string> split_string(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

};
