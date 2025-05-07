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
#include <string>
#include <filesystem>

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "pos/include/common.h"
#include "pos/include/handle.h"
#include "pos/include/oob.h"
#include "pos/include/oob/ckpt_predump.h"
#include "pos/include/utils/string.h"
#include "pos/include/utils/system.h"
#include "pos/include/utils/command_caller.h"
#include "pos/cli/cli.h"


pos_retval_t handle_predump(pos_cli_options_t &clio){
    pos_retval_t retval = POS_SUCCESS;
    oob_functions::cli_ckpt_predump::oob_call_data_t call_data;

    std::string mount_cmd, mount_result;
    std::uintmax_t nb_removed_files;
    bool has_mount_before = false;
    uint64_t total_mem_bytes, avail_mem_bytes;
    std::string mount_existance_file;
    std::ofstream mount_existance_file_stream;

    std::string criu_cmd, criu_result;
    std::thread criu_thread;
    std::promise<pos_retval_t> criu_thread_promise;
    std::future<pos_retval_t> criu_thread_future = criu_thread_promise.get_future();

    validate_and_cast_args(
        /* clio */ clio, 
        /* rules */ {
            {
                /* meta_type */ kPOS_CliMeta_Pid,
                /* meta_name */ "pid",
                /* meta_desp */ "pid of the process to be migrated",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    clio.metas.ckpt.pid = std::stoull(meta_val);
                exit:
                    return retval;
                },
                /* is_required */ true
            },
            {
                /* meta_type */ kPOS_CliMeta_Dir,
                /* meta_name */ "dir",
                /* meta_desp */ "directory to store the checkpoint files",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    std::filesystem::path absolute_path;

                    absolute_path = std::filesystem::absolute(meta_val);

                    if(absolute_path.string().size() >= oob_functions::cli_ckpt_predump::kCkptFilePathMaxLen){
                        POS_WARN(
                            "ckpt file path too long: given(%lu), expected_max(%lu)",
                            absolute_path.string().size(),
                            oob_functions::cli_ckpt_predump::kCkptFilePathMaxLen
                        );
                        retval = POS_FAILED_INVALID_INPUT;
                        goto exit;
                    }

                    memset(clio.metas.ckpt.ckpt_dir, 0, oob_functions::cli_ckpt_predump::kCkptFilePathMaxLen);
                    memcpy(clio.metas.ckpt.ckpt_dir, absolute_path.string().c_str(), absolute_path.string().size());

                exit:
                    return retval;
                },
                /* is_required */ true
            },
            {
                /* meta_type */ kPOS_CliMeta_SkipTarget,
                /* meta_name */ "skip-target",
                /* meta_desp */ "resource types to skip dumpping",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    uint64_t i;
                    std::vector<std::string> substrings;
                    std::string substring;
                    typename std::map<pos_resource_typeid_t, std::string>::iterator map_iter;
                    bool found_resource = false;

                    substrings = POSUtil_String::split_string(meta_val, ',');

                    clio.metas.ckpt.nb_skip_targets = 0;
                    for(i=0; i<substrings.size(); i++){
                        substring = substrings[i];
                        found_resource = false;
                        for(map_iter = pos_resource_map.begin(); map_iter != pos_resource_map.end(); map_iter++){
                            if(map_iter->second == substring){
                                found_resource = true;
                                clio.metas.ckpt.skip_targets[clio.metas.ckpt.nb_skip_targets] = map_iter->first;
                                clio.metas.ckpt.nb_skip_targets += 1;
                                POS_ASSERT(clio.metas.ckpt.nb_skip_targets <= oob_functions::cli_ckpt_predump::kSkipTargetMaxNum);
                                break;
                            }
                        }
                        if(unlikely(found_resource == false)){
                            POS_WARN("unrecognized resource type %s", substring.c_str());
                            retval = POS_FAILED_INVALID_INPUT;
                            goto exit;
                        }
                    }

                exit:
                    return retval;
                },
                /* is_required */ false
            },
            {
                /* meta_type */ kPOS_CliMeta_Target,
                /* meta_name */ "target",
                /* meta_desp */ "resource types to do dumpping",
                /* cast_func */ [](pos_cli_options_t &clio, std::string& meta_val) -> pos_retval_t {
                    pos_retval_t retval = POS_SUCCESS;
                    uint64_t i;
                    std::vector<std::string> substrings;
                    std::string substring;
                    typename std::map<pos_resource_typeid_t, std::string>::iterator map_iter;
                    bool found_resource = false;

                    substrings = POSUtil_String::split_string(meta_val, ',');

                    clio.metas.ckpt.nb_targets = 0;
                    for(i=0; i<substrings.size(); i++){
                        substring = substrings[i];
                        found_resource = false;
                        for(map_iter = pos_resource_map.begin(); map_iter != pos_resource_map.end(); map_iter++){
                            if(map_iter->second == substring){
                                found_resource = true;
                                clio.metas.ckpt.targets[clio.metas.ckpt.nb_targets] = map_iter->first;
                                clio.metas.ckpt.nb_targets += 1;
                                POS_ASSERT(clio.metas.ckpt.nb_targets <= oob_functions::cli_ckpt_predump::kTargetMaxNum);
                                break;
                            }
                        }
                        if(unlikely(found_resource == false)){
                            POS_WARN("unrecognized resource type %s", substring.c_str());
                            retval = POS_FAILED_INVALID_INPUT;
                            goto exit;
                        }
                    }

                exit:
                    return retval;
                },
                /* is_required */ false
            },
        },
        /* collapse_rule */ [](pos_cli_options_t& clio) -> pos_retval_t {
            pos_retval_t retval = POS_SUCCESS;

            if(unlikely(clio.metas.ckpt.nb_targets > 0 && clio.metas.ckpt.nb_skip_targets > 0)){
                POS_WARN(
                    "you can't specified both the whitelist and blacklist of resource types to pre-dump (use either '--target' or '--skip-target')"
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }

            if(clio.metas.ckpt.nb_targets == 0 && clio.metas.ckpt.nb_skip_targets == 0){
                POS_WARN("no target and skip-target specified, default to pre-dump only stateful resources (e.g., device memory)");
            }

        exit:
            return retval;
        }
    );

    mount_existance_file = std::string(clio.metas.ckpt.ckpt_dir) + std::string("/tmpfs_mount.lock");

    // step 1: make sure the directory exist and fresh
    if (std::filesystem::exists(clio.metas.ckpt.ckpt_dir)) {
        try {
            if(std::filesystem::exists(mount_existance_file)){
                has_mount_before = true;
            }
            nb_removed_files = 0;
            for(auto& de : std::filesystem::directory_iterator(clio.metas.ckpt.ckpt_dir)) {
                // returns the number of deleted entities since c++17:
                nb_removed_files += std::filesystem::remove_all(de.path());
            }
            POS_LOG(
                "clean old assets under specified pre-dump dir: dir(%s), nb_removed_files(%lu)",
                clio.metas.ckpt.ckpt_dir, nb_removed_files
            );
            POS_LOG("reuse pre-dump dir: %s",  clio.metas.ckpt.ckpt_dir);
        } catch (const std::exception& e) {
            POS_WARN(
                "failed to remove old assets under specified pre-dump dir: dir(%s), error(%s)",
                clio.metas.ckpt.ckpt_dir, e.what()
            );
            retval = POS_FAILED;
            goto exit;
        }
    } else {
        try {
            std::filesystem::create_directories(clio.metas.ckpt.ckpt_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            POS_WARN(
                "failed to create pre-dump directory: dir(%s), error(%s)",
                clio.metas.ckpt.ckpt_dir, e.what()
            );
            retval = POS_FAILED;
            goto exit;
        }
    }

    // step 2: mount the memory to tmpfs
    if(has_mount_before == false){
        // obtain available memory on the system
        retval = POSUtilSystem::get_memory_info(total_mem_bytes, avail_mem_bytes);
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed predump, failed to obtain memory information of the ststem");
            retval = POS_FAILED;
            goto exit;
        }
        if(unlikely(avail_mem_bytes <= MB(128))){
            POS_WARN(
                "failed predump, not enough memory on the system: total(%lu bytes), avail(%lu bytes)",
                total_mem_bytes, avail_mem_bytes
            );
            retval = POS_FAILED;
            goto exit;
        }

        // execute mount cmd
        mount_cmd   = std::string("mount -t tmpfs -o size=")
                    + POSUtilSystem::format_byte_number(avail_mem_bytes * 0.8)
                    + std::string(" tmpfs ") + std::string(clio.metas.ckpt.ckpt_dir);
        
        retval = POSUtil_Command_Caller::exec_sync(
            mount_cmd,
            mount_result,
            /* ignore_error */ false,
            /* print_stdout */ true,
            /* print_stderr */ true
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("failed to mount predump directory to tmpfs, the predump might be slowed down due to storage IO");
        } else {
            POS_LOG(
                "mount pre-dump dir to tmpfs: size(%lu), dir(%s)",
                avail_mem_bytes * 0.8, clio.metas.ckpt.ckpt_dir
            );
        }
    }

    // step 3: create mount existance file
    POS_ASSERT(!std::filesystem::exists(mount_existance_file));
    mount_existance_file_stream.open(mount_existance_file);
    if(unlikely(!mount_existance_file_stream.is_open())){
        POS_WARN(
            "failed to create mount existance file, yet still successfully mount to tmpfs: path(%s)",
            mount_existance_file.c_str()
        );
    }
    mount_existance_file_stream << std::to_string(static_cast<int>(avail_mem_bytes * 0.8));
    mount_existance_file_stream.close();

    // step 4: check whether CPU-side (CRIU) support predump
    criu_cmd = std::string("criu check --all");
    criu_result.clear();
    retval = POSUtil_Command_Caller::exec_sync(
        criu_cmd,
        criu_result,
        /* ignore_error */ true,
        /* print_stdout */ false,
        /* print_stderr */ false
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN("predump failed, failed to check criu supports: retval(%s)", criu_result.c_str());
        goto exit;
    }
    if(criu_result.find("Dirty tracking is OFF. Memory snapshot will not work.") != std::string::npos){
        POS_WARN("CRIU doesn't support pre-dump, please consider recompile your kernel, see https://github.com/checkpoint-restore/criu/issues/545#issuecomment-416024221");
        retval = POS_FAILED;
        goto exit;
    }

    // step 5: CPU-side predump (async)
    criu_cmd    = std::string("criu pre-dump ")
                + std::string("--tree ") + std::to_string(clio.metas.ckpt.pid) + std::string (" ")
                + std::string("--images-dir ") + std::string(clio.metas.ckpt.ckpt_dir) + std::string (" ")
                + std::string("--leave-running --track-mem --shell-job --display-stats");
    criu_result.clear();
    retval = POSUtil_Command_Caller::exec_async(
        criu_cmd, criu_thread, criu_thread_promise, criu_result,
        /* ignore_error */ false,
        /* print_stdout */ false,
        /* print_stderr */ false
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN("predump failed, failed to start cpu-side predump thread: retval(%u)", retval);
        goto exit;
    }

    // step 6: GPU-side predump (sync)
    call_data.pid = clio.metas.ckpt.pid;
    memcpy(
        call_data.ckpt_dir,
        clio.metas.ckpt.ckpt_dir,
        oob_functions::cli_ckpt_predump::kCkptFilePathMaxLen
    );
    memcpy(
        call_data.targets,
        clio.metas.ckpt.targets,
        sizeof(call_data.targets)
    );
    memcpy(
        call_data.skip_targets,
        clio.metas.ckpt.skip_targets,
        sizeof(call_data.skip_targets)
    );
    call_data.nb_targets = clio.metas.ckpt.nb_targets;
    call_data.nb_skip_targets = clio.metas.ckpt.nb_skip_targets;
    retval = clio.local_oob_client->call(kPOS_OOB_Msg_CLI_Ckpt_PreDump, &call_data);
    if(POS_SUCCESS != call_data.retval){
        POS_WARN("predump failed, gpu-side predump failed: %s", call_data.retmsg);
        goto exit;
    }

    // step 7: check CPU-side predump result
    if(criu_thread.joinable()){ criu_thread.join(); }
    retval = criu_thread_future.get();
    if(POS_SUCCESS != retval){
        POS_WARN("predump failed, cpu-side predump failed: %s", criu_result.c_str());
        goto exit;
    }

    POS_LOG("predump done");

 exit:
    return retval;
}
