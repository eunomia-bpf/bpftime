#ifndef _BPFTIME_GLOBAL_SHM_HPP
#define _BPFTIME_GLOBAL_SHM_HPP

namespace bpftime
{
namespace shm_common
{
constexpr const char *DEFAULT_GLOBAL_SHM_NAME = "bpftime_maps_shm";
constexpr const char *DEFAULT_GLOBAL_HANDLER_NAME = "bpftime_handler";
constexpr const char *DEFAULT_SYSCALL_PID_SET_NAME = "bpftime_syscall_pid_set";
constexpr const char *DEFAULT_AGENT_CONFIG_NAME = "bpftime_agent_config";

const char *get_global_shm_name();

} // namespace shm_common
} // namespace bpftime

#endif
