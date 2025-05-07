#include <iostream>
#include <string>
#include <sstream>

#include "pos/include/common.h"
#include "pos/include/log.h"
#include "pos/cli/cli.h"

pos_retval_t handle_help(pos_cli_options_t &clio){
    std::stringstream helper_message_shell;
    std::stringstream helper_message_help, helper_message_start;
    std::stringstream helper_message_pre_dump, helper_message_dump, helper_message_restore, helper_message_pre_restore, helper_message_clean;
    std::stringstream helper_message_migration;
    std::stringstream helper_message_trace;

    helper_message_help 
        << "--help:                  print help message (like you just did)\n"
        << "    --target <action>    [optional] print helper message for a specific action\n"
        << "\n"
        << "    e.g., pos_cli --help --target=pre-dump\n";

    helper_message_start
        << "--start:                 start specified PhOS component\n"
        << "    --target <component> specified component to be started\n"
        << "\n"
        << "    e.g., for started PhOS daemon, run 'pos_cli --start --target daemon'\n";

    helper_message_pre_dump 
        << "--pre-dump:                 pre-dump the state of specified GPU process\n"
        << "     --pid <pid>            PID of the process to be pre-dumped\n"
        << "     --dir <dir>            directory to store the pre-dumped state\n"
        << "     --target <str>         [optional] names of the resource to be pre-dumped, splited using ','\n"
        << "     --skip-target <str>    [optional] names of the resource NOT to be pre-dumped, splited using ','\n"
        << "\n"
        << "     for both 'target' and 'skip-target', supported resource names includes\n"
        << "        - \"cuda_context\"\n"
        << "        - \"cuda_module\"\n"
        << "        - \"cuda_function\"\n"
        << "        - \"cuda_var\"\n"
        << "        - \"cuda_device\"\n"
        << "        - \"cuda_memory\"\n"
        << "        - \"cuda_stream\"\n"
        << "        - \"cuda_event\"\n"
        << "\n"
        << "     e.g., 'pos_cli --pre-dump --dir=./ckpt --pid=14392 --target=cuda_memory,cuda_stream\n";

    helper_message_dump     
        << "--dump:                     dump the state of specified GPU process\n"
        << "     --pid <pid>            PID of the process to be dumped\n"
        << "     --dir <dir>            directory to store the dumped state\n"
        << "     --target <str>         [optional] names of the resource to be dumped, splited using ','\n"
        << "     --skip-target <str>    [optional] names of the resource NOT to be dumped, splited using ','\n"
        << "\n"
        << "     for both 'target' and 'skip-target', supported resource names includes\n"
        << "        - \"cuda_context\"\n"
        << "        - \"cuda_module\"\n"
        << "        - \"cuda_function\"\n"
        << "        - \"cuda_var\"\n"
        << "        - \"cuda_device\"\n"
        << "        - \"cuda_memory\"\n"
        << "        - \"cuda_stream\"\n"
        << "        - \"cuda_event\"\n"
        << "\n"
        << "     e.g., 'pos_cli --dump --dir=./ckpt --pid=14392 --target=cuda_memory,cuda_stream\n";

    helper_message_restore  
        << "--restore:                  restore the state of specified GPU process\n"
        << "     --dir <dir>            directory that stores the previously dumped state\n"
        << "\n"
        << "     e.g., 'pos_cli --restore --dir=./ckpt\n";


    helper_message_pre_restore 
        << "--pre-restore:              pre-restore specific GPU resources in advance (e.g., CUmodules)\n"
        << "     --target <str>         names of the resources to be dumped to be pre-restored\n"
        << "     --dir <dir>            directory to that stores the dumped state\n"
        << "\n"
        << "     for 'target', supported resource names includes\n"
        << "        - \"cuda_context\"\n"
        << "        - \"cuda_module\"\n"
        << "        - \"cuda_function\"\n"
        << "        - \"cuda_var\"\n"
        << "        - \"cuda_device\"\n"
        << "        - \"cuda_memory\"\n"
        << "        - \"cuda_stream\"\n"
        << "        - \"cuda_event\"\n"
        << "\n"
        << "     e.g., 'pos_cli --pre-restore --target=cuda_module,cuda_function --dir=./ckpt\n";

    helper_message_clean
        << "--clean:                    clean speficified checkpoint dumped files\n"
        << "     --dir <dir>            directory to that stores the dumped state\n"
        << "\n"
        << "     e.g., 'pos_cli --clean --dir=./ckpt\n";

    helper_message_migration
        << "--migrate:              migrate the state of specified GPU process to another machine\n"
        << "    TODO\n";

    helper_message_trace
        << "--trace-resource:       trace the resource touch behaviour of the GPU program\n"
        << "    --subaction <act>   subaction to control the trace behaviour, either 'start' or 'stop'"
        << "     --pid <pid>        PID of the process to be traced\n"
        << "\n"
        << "     e.g., for starting trace, 'pos_cli --trace-resource --subaction=start --pid=23491'\n";

    helper_message_shell    << "FORMAT: pos_cli --ACTION [--METADATA --VALUE]\n"
                            << "\n"
                            << "[A. Miscellaneous]\n"
                            << "------------------------------------------------------------------------------------\n"
                                << helper_message_help.str()
                                << "\n"
                                << helper_message_start.str()
                            << "------------------------------------------------------------------------------------\n"
                            << "\n\n"
                            << "[B. Checkpoint / Restore]\n"
                            << "------------------------------------------------------------------------------------\n"
                                << helper_message_pre_dump.str()
                                << "\n"
                                << helper_message_dump.str()
                                << "\n"
                                << helper_message_restore.str()
                                << "\n"
                                << helper_message_pre_restore.str()
                                << "\n"
                                << helper_message_clean.str()
                            << "------------------------------------------------------------------------------------\n"
                            << "\n\n"
                            << "[C. Migration]\n"
                            << "------------------------------------------------------------------------------------\n"
                                << helper_message_migration.str()
                            << "------------------------------------------------------------------------------------\n"
                            << "\n\n"
                            << "[C. Trace]\n"
                            << "------------------------------------------------------------------------------------\n"
                                << helper_message_trace.str()
                            << "------------------------------------------------------------------------------------\n"
                            << "\n\n"
                            ;

    POS_LOG(
        ">>>>>>>>>> PhOS Client Interface <<<<<<<<<<\n%s\n\n%s",
        pos_banner.c_str(), helper_message_shell.str().c_str()
    );

    return POS_SUCCESS;
}
