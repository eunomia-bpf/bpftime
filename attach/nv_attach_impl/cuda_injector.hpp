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
	CUcontext cuda_ctx{ nullptr };

	// Storing a backup of code, for illustration.
	// You can remove or adapt this if you don't actually need code
	// injection.
	struct CodeBackup {
		CUdeviceptr addr;
	};
	std::vector<CodeBackup> backups;
	std::string orig_ptx;
	pos_cli_options_t clio_checkpoint;
	pos_cli_options_t clio_restore;
	explicit CUDAInjector(pid_t pid, std::string orig_ptx);

	bool attach();

	bool detach();

    private:
	// ------------------------------------------------------------------------
	// Below is minimal logic to demonstrate how you MIGHT find a CUDA
	// context. In reality, hooking into a remote process's memory for CUDA
	// contexts is significantly more complex (symbol lookup, driver calls,
	// etc.).
	// ------------------------------------------------------------------------
	bool get_cuda_context();

	bool validate_cuda_context(CUcontext remote_ctx);

    public:
	// Demonstrates how you might inject PTX or backup/restore code on the
	// fly in a remote context. This is a stub for illustration.
	bool inject_ptx(const char *ptx_code1, CUdeviceptr target_addr,
			size_t code_size, CUmodule &module);
};
} // namespace attach
} // namespace bpftime

#endif
