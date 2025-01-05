#ifndef _CUDA_INJECTOR_HPP
#define _CUDA_INJECTOR_HPP

#include "cuda.h"
#include "pos/cli/cli.h"
#include "pos/include/workspace.h"
#include "pos/include/common.h"
#include "pos/include/oob.h"
#include "pos/include/oob/ckpt_dump.h"
#include "pos/include/oob/restore.h"
#include <string>
#include <vector>
namespace bpftime
{
namespace attach
{
// ----------------------------------------------------------------------------
// A simple wrapper class to handle attaching to a CUDA context in another
// process. In a real scenario, you might separate this into its own .hpp/.cpp
// files.
// ----------------------------------------------------------------------------
class CUDAInjector {
    public:
	pid_t target_pid;
	pos_cli_options_t clio_checkpoint;
	pos_cli_options_t clio_restore;
	explicit CUDAInjector(pid_t pid);
    public:
	// Demonstrates how you might inject PTX or backup/restore code on the
	// fly in a remote context. This is a stub for illustration.
	bool inject_ptx();
};
} // namespace attach
} // namespace bpftime

#endif
