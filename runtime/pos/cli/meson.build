# Copyright 2024 The PhoenixOS Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# >>> project definition
project('PhoenixOS_CLI', ['CPP'],
	# Get version number from file.
	version: run_command(find_program('cat'),files('../../VERSION'), check: true).stdout().strip(),
)
project_name = 'phoenix_os_cli'
project_name_abbreviation = 'pos_cli'

add_project_arguments('-O0',  language : 'cpp')

scan_src_path = meson.current_source_dir() + '/../../scripts/utils/glob_src.py'

# >>>>>>>>>>>>>> build prepare >>>>>>>>>>>>>>
# get environment variables of the build process
env = environment()

# we use pkg-config to collect dependecies
pkgconfig = find_program('pkg-config')
bash = find_program('bash')

# path of built libraries by PhOS build system
lib_path = meson.current_source_dir() + '/../../lib'

# compile flags
c_args = []

# load flags
ld_args = [ '-L'+lib_path ]

# include directories
inc_dirs = [ '../../lib', '../../lib/pos/include', '../../' ]

# source files
sources = []

# >>>>>>>>>>>>>> setup configurations >>>>>>>>>>>>>>


# >>>>>> [1] platform configs >>>>>>>>
# root directory of the project
conf_platform_project_root = run_command('sh', '-c', 'echo $POS_BUILD_CONF_PlatformProjectRoot').stdout().strip()
if conf_platform_project_root == ''
    assert(false, 'no project root directory was set')
endif
# >>>>>>>> [1] platform configs >>>>>>>>


# >>>>>>>> [2] runtime configs >>>>>>>>
# target selection (options: cuda, todos: rocm/ascend)
conf_runtime_target = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeTarget').stdout().strip()
if conf_runtime_target != 'cuda'
    assert(false, 'runtime target ' + conf_runtime_target + ' is currently not supported')
endif

# target version (example: "11.3" for CUDA 11.3)
conf_runtime_target_version = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeTargetVersion').stdout().strip()
if conf_runtime_target_version == ''
    assert(false, 'no runtime target version was provided')
endif

# whether to print all fatal messages
conf_runtime_enable_print_error = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnablePrintError').stdout().strip().to_int()
if conf_runtime_enable_print_error != 0 and conf_runtime_enable_print_error != 1
    assert(
        false, 
        'conf_runtime_enable_print_error get invalid value: ' + conf_runtime_enable_print_error.to_string()
    )
endif

# whether to print all warning messages
conf_runtime_enable_print_warn = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnablePrintWarn').stdout().strip().to_int()
if conf_runtime_enable_print_warn != 0 and conf_runtime_enable_print_warn != 1
    assert(
        false,
        'conf_runtime_enable_print_warn get invalid value: ' + conf_runtime_enable_print_warn.to_string()
    )
endif

# whether to print all log messages
conf_runtime_enable_print_log = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnablePrintLog').stdout().strip().to_int()
if conf_runtime_enable_print_log != 0 and conf_runtime_enable_print_log != 1
    assert(
        false, 
        'conf_runtime_enable_print_log get invalid value: ' + conf_runtime_enable_print_log.to_string()
    )
endif

# whether to print all debug messages
conf_runtime_enable_print_debug = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnablePrintDebug').stdout().strip().to_int()
if conf_runtime_enable_print_debug != 0 and conf_runtime_enable_print_debug != 1
    assert(
        false, 
        'conf_runtime_enable_print_debug get invalid value: ' + conf_runtime_enable_print_debug.to_string()
    )
endif

# whether to print fatal/warn/debug/log with colors
# should be disabled when try to dump all outputs to a file
conf_runtime_enable_print_with_color = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnablePrintWithColor').stdout().strip().to_int()
if conf_runtime_enable_print_with_color != 0 and conf_runtime_enable_print_with_color != 1
    assert(
        false, 
        'conf_runtime_enable_print_with_color get invalid value: ' + conf_runtime_enable_print_with_color.to_string()
    )
endif

# whether to check correctness of function parameters, return values, etc.,
# which might cause extra runtime burden, should only be enabled while debuging
conf_runtime_enable_debug_check = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnableDebugCheck').stdout().strip().to_int()
if conf_runtime_enable_debug_check != 0 and conf_runtime_enable_debug_check != 1
    assert(
        false,
        'conf_runtime_enable_debug_check get invalid value: ' + conf_runtime_enable_debug_check.to_string()
    )
endif

# whether to check whether POS finish the hijacking logic of called APIs,
# which might cause extra runtime burden, should only be enabled while debuging
conf_runtime_enable_hijack_api_check = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnableHijackApiCheck').stdout().strip().to_int()
if conf_runtime_enable_hijack_api_check != 0 and conf_runtime_enable_hijack_api_check != 1
    assert(
        false, 
        'conf_runtime_enable_hijack_api_check get invalid value: ' + conf_runtime_enable_hijack_api_check.to_string()
    )
endif

# whether to trace the statistics
conf_runtime_enable_trace = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnableTrace').stdout().strip().to_int()
if conf_runtime_enable_trace != 0 and conf_runtime_enable_trace != 1
    assert(
        false, 
        'conf_runtime_enable_trace get invalid value: ' + conf_runtime_enable_trace.to_string()
    )
endif

# whether to trace the memory statistics
conf_runtime_enable_memory_trace = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeEnableMemoryTrace').stdout().strip().to_int()
if conf_runtime_enable_memory_trace != 0 and conf_runtime_enable_memory_trace != 1
    assert(
        false, 
        'conf_runtime_enable_memory_trace get invalid value: ' + conf_runtime_enable_memory_trace.to_string()
    )
endif

# log path of PhOS daemon
conf_runtime_default_daemon_log_path = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeDefaultDaemonLogPath').stdout().strip()
if conf_runtime_default_daemon_log_path == ''
    assert(false, 'no default log path of PhOS daemon is provided')
endif

# log path of PhOS client
conf_runtime_default_client_log_path = run_command('sh', '-c', 'echo $POS_BUILD_CONF_RuntimeDefaultClientLogPath').stdout().strip()
if conf_runtime_default_client_log_path == ''
    assert(false, 'no default log path of PhOS client is provided')
endif
# >>>>>>>> [2] runtime configs >>>>>>>>


# >>>>>>>> [3] evaluation configs >>>>>>>>
# checkpoint optimization level
#   - 0: no ckpt support
#   - 1: naive ckpt support
#   - 2: pos ckpt support
conf_eval_ckpt_opt_level = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalCkptOptLevel').stdout().strip().to_int()
if conf_eval_ckpt_opt_level != 0 and conf_eval_ckpt_opt_level != 1 and conf_eval_ckpt_opt_level != 2
    assert(false, 'conf_eval_ckpt_opt_level with level ' + conf_eval_ckpt_opt_level.to_string() + ' is currently not supported')
endif

# sync/async checkpoint (level 1,2) optimization -> flag to control increamental checkpoint
conf_eval_ckpt_enable_increamental = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalCkptEnableIncremental').stdout().strip().to_int()
if conf_eval_ckpt_enable_increamental != 0 and conf_eval_ckpt_enable_increamental != 1
    assert(
        false,
        'conf_eval_ckpt_enable_increamental get invalid value: ' + conf_eval_ckpt_enable_increamental.to_string()
    )
endif

# async checkpoint (level 2) optimization -> flag to control checkpoint pipeline
conf_eval_ckpt_enable_pipeline = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalCkptEnablePipeline').stdout().strip().to_int()
if conf_eval_ckpt_enable_pipeline != 0 and conf_eval_ckpt_enable_pipeline != 1
    assert(
        false,
        'conf_eval_ckpt_enable_pipeline get invalid value: ' + conf_eval_ckpt_enable_pipeline.to_string()
    )
endif

# default continuous checkpoint interval (unit in ms)
conf_eval_default_ckpt_interval_ms = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalCkptDefaultIntervalMs').stdout().strip().to_int()
if conf_eval_default_ckpt_interval_ms < 0
    assert(
        false,
        'conf_eval_default_ckpt_interval_ms get invalid value: ' + conf_eval_default_ckpt_interval_ms.to_string()
    )
endif

# migration optimization level
#   - 0: no migration support
#   - 1: naive migration support
#   - 2: pos migration support
conf_eval_migr_opt_level = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalMigrOptLevel').stdout().strip().to_int()
if conf_eval_migr_opt_level != 0 and conf_eval_migr_opt_level != 1 and conf_eval_migr_opt_level != 2
    assert(false, 'conf_eval_migr_opt_level with level ' + conf_eval_migr_opt_level.to_string() + ' is currently not supported')
endif

# whether to enable context pool to restore
conf_eval_rst_enable_context_pool = run_command('sh', '-c', 'echo $POS_BUILD_CONF_EvalRstEnableContextPool').stdout().strip().to_int()
if conf_eval_rst_enable_context_pool != 0 and conf_eval_rst_enable_context_pool != 1 and conf_eval_rst_enable_context_pool != 2
    assert(false, 'conf_eval_rst_enable_context_pool with level ' + conf_eval_rst_enable_context_pool.to_string() + ' is currently not supported')
endif
# >>>>>>>> [3] evaluation configs >>>>>>>>


# add common headers
subdir('pos/include')


# >>>>>>>>>>>>>> find target libraries >>>>>>>>>>>>>>
if conf_runtime_target == 'cuda'
    # cuda_version = run_command(bash, '../scripts/utils/get_cuda_version.sh').stdout().strip()
    cuda_version = conf_runtime_target_version
    cuda_pc_path = '/usr/lib/pkgconfig'
    cuda_modules = ['cublas', 'cuda', 'cudart', 'cufft', 'cufftw', 'cuinj64', 'curand', 'cusolver', 'cusparse']
    founded_cuda_modules = []
    message('>>> Detecting CUDA toolkit, assume:')
    message('>>>>>> CUDA version:               ' + cuda_version)
    message('>>>>>> Path of pkg-config files:   ' + cuda_pc_path)

    env.set('PKG_CONFIG_PATH', cuda_pc_path)
    foreach cuda_module : cuda_modules
        cuda_module_cflags = run_command(pkgconfig, '--cflags', cuda_module+'-'+cuda_version, env: env, check: false)
        cuda_module_ldflags = run_command(pkgconfig, '--libs', '--static', cuda_module+'-'+cuda_version, env: env, check: false) # add '--static' option if static link is needed
        cuda_module_version = run_command(pkgconfig, '--modversion', cuda_module+'-'+cuda_version, env: env, check: false)
        if cuda_module_cflags.returncode() != 0 or cuda_module_ldflags.returncode() != 0 or cuda_module_version.returncode() != 0
            message('>>>>>> Failed to find ' + cuda_module+'-'+cuda_version)
        else
            founded_cuda_modules += cuda_module
            c_args += cuda_module_cflags.stdout().split()
            ld_args += cuda_module_ldflags.stdout().split()
            message('>>>>>> Found ' + cuda_module + ', version is ' + cuda_module_version.stdout().split()[0])
        endif
    endforeach
    assert(
        founded_cuda_modules.length() == cuda_modules.length(),
        'Only find ' + founded_cuda_modules.length().to_string() + ' CUDA modules, expected ' + cuda_modules.length().to_string()
    )
    message('>>> Detecting CUDA toolkit done\n')
endif


# >>>>>>>>>>>>>> setup sources, libraries and includes >>>>>>>>>>>>>>
sources += run_command('python3', files(scan_src_path), './src', check: false).stdout().strip().split('\n')
message()
ld_args += [ '-pthread' ]
ld_args += [ '-lclang', '-lyaml-cpp', '-lpos' ]
ld_args += ['-lprotobuf', '-lprotobuf-lite', '-lprotoc']    


# >>>>>>>>>>>>>> setup build options >>>>>>>>>>>>>>
c_args += ['--std=c++2a']

executable(
    project_name_abbreviation,
    sources,
    cpp_args: c_args,
	link_args: ld_args,
	include_directories: inc_dirs,
	install: false
)
