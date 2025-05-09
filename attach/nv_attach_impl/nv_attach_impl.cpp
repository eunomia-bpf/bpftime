#include "nv_attach_impl.hpp"
#include "cuda_injector.hpp"
#include "frida-gum.h"

#include "pos/include/common.h"
#include "spdlog/spdlog.h"
#include <asm/unistd.h> // For architecture-specific syscall numbers

#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/uio.h>
#include <link.h>
#include <pos/cuda_impl/utils/fatbin.h>
#include <unistd.h>
#include <vector>
#include <nvrtc.h>
using namespace bpftime;
using namespace attach;

#define NVRTC_SAFE_CALL(x)                                                     \
	do {                                                                   \
		nvrtcResult result = x;                                        \
		if (result != NVRTC_SUCCESS) {                                 \
			SPDLOG_ERROR("NVRTC ERROR: {} at {}:{}",               \
				     nvrtcGetErrorString(result), __FILE__,    \
				     __LINE__);                                \
			throw std::runtime_error("nvrtc error");               \
		}                                                              \
	} while (0)

typedef struct _CUDARuntimeFunctionHooker {
	GObject parent;
} CUDARuntimeFunctionHooker;

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data);

// #define EXAMPLE_TYPE_LISTENER (cuda_runtime_function_hooker_iface_init())
G_DECLARE_FINAL_TYPE(CUDARuntimeFunctionHooker, cuda_runtime_function_hooker,
		     BPFTIME, NV_ATTACH_IMPL, GObject)
G_DEFINE_TYPE_EXTENDED(
	CUDARuntimeFunctionHooker, cuda_runtime_function_hooker, G_TYPE_OBJECT,
	0,
	G_IMPLEMENT_INTERFACE(GUM_TYPE_INVOCATION_LISTENER,
			      cuda_runtime_function_hooker_iface_init))

extern "C" {

typedef struct __attribute__((__packed__)) fat_elf_header {
	uint32_t magic;
	uint16_t version;
	uint16_t header_size;
	uint64_t size;
} fat_elf_header_t;
}

static void example_listener_on_enter(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	auto gum_ctx = gum_interceptor_get_current_invocation();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
		SPDLOG_INFO("Mocking __cudaRegisterFatBinary..");

		auto header = (__fatBinC_Wrapper_t *)
			gum_invocation_context_get_nth_argument(gum_ctx, 0);
		auto data = (const char *)header->data;
		fat_elf_header_t *curr_header = (fat_elf_header_t *)data;
		const char *tail = (const char *)curr_header;
		while (true) {
			if (curr_header->magic == FATBIN_TEXT_MAGIC) {
				SPDLOG_INFO(
					"Got CUBIN section header size = {}, size = {}",
					static_cast<int>(
						curr_header->header_size),
					static_cast<int>(curr_header->size));
				tail = ((const char *)curr_header) +
				       curr_header->header_size +
				       curr_header->size;
				curr_header = (fat_elf_header_t *)tail;
			} else {
				break;
			}
		};
		std::vector<char> data_vec(data, tail);
		SPDLOG_INFO("Finally size = {}", data_vec.size());

		std::vector<std::string> ptx_out;
		{
			SPDLOG_INFO("Listing functions in the patched ptx");
			std::vector<POSCudaFunctionDesp *> desp;
			std::map<std::string, POSCudaFunctionDesp *> cache;

			auto result = POSUtil_CUDA_Fatbin::
				obtain_functions_from_cuda_binary(
					(uint8_t *)data_vec.data(),
					data_vec.size(), &desp, cache, ptx_out);
			if (result != POS_SUCCESS) {
				SPDLOG_ERROR(
					"Unable to parse functions from patched fatbin: {}",
					(int)result);
				return;
			}
			SPDLOG_INFO(
				"Got these functions in the patched fatbin");
			for (const auto &item : desp) {
				SPDLOG_INFO("{}", item->signature);
			}
		}
		if (ptx_out.size() != 1) {
			SPDLOG_ERROR(
				"Expect the loaded fatbin to contain only 1 PTX code section, but it contains {}",
				ptx_out.size());
			return;
		}

		/**
		Here we can patch the PTX. Then recompile it.
		*/

		SPDLOG_INFO("Recompiling PTX with nvcc..");
		char tmp_dir[] = "/tmp/bpftime-recompile-nvcc.XXXXXX";
		mkdtemp(tmp_dir);
		std::filesystem::path work_dir(tmp_dir);
		SPDLOG_INFO("Working directory: {}", work_dir.c_str());
		std::string command = "nvcc ";
		{
			auto ptx_in = work_dir / "main.ptx";
			SPDLOG_INFO("PTX IN: {}", ptx_in.c_str());
			std::ofstream ofs(ptx_in);
			ofs << ptx_out[0];
			command += ptx_in;
			command += " ";
		}
		command += "-fatbin ";
		auto fatbin_out = work_dir / "out.fatbin";
		command += "-o ";
		command += fatbin_out;
		SPDLOG_INFO("Fatbin out {}", fatbin_out.c_str());
		SPDLOG_INFO("Starting nvcc: {}", command);
		if (int err = system(command.c_str()); err != 0) {
			SPDLOG_ERROR("Unable to execute nvcc");
			return;
		}
		SPDLOG_INFO("NVCC execution done.");
		std::vector<uint8_t> fatbin_out_buf;
		{
			std::ifstream ifs(fatbin_out,
					  std::ios::binary | std::ios::ate);
			auto file_tail = ifs.tellg();
			ifs.seekg(0, std::ios::beg);

			fatbin_out_buf.resize(file_tail);
			ifs.read((char *)fatbin_out_buf.data(), file_tail);
		}

		SPDLOG_INFO("Got patched fatbin in {} bytes",
			    fatbin_out_buf.size());
		auto patched_fatbin_ptr =
			std::make_unique<std::vector<uint8_t>>(fatbin_out_buf);
		auto patched_header = std::make_unique<__fatBinC_Wrapper_t>();
		auto patched_header_ptr = patched_header.get();
		patched_header->magic = 0x466243b1;
		patched_header->version = 1;
		patched_header->data =
			(const unsigned long long *)patched_fatbin_ptr->data();
		patched_header->filename_or_fatbins = 0;
		context->impl->stored_binaries_body.push_back(
			std::move(patched_fatbin_ptr));
		context->impl->stored_binaries_header.push_back(
			std::move(patched_header));
		// Set the patched header as the argument
		gum_invocation_context_replace_nth_argument(gum_ctx, 0,
							    patched_header_ptr);
	}
}

static void example_listener_on_leave(GumInvocationListener *listener,
				      GumInvocationContext *ic)
{
	SPDLOG_INFO("On leave");
	// injector.detach();
	auto context =
		GUM_IC_GET_FUNC_DATA(ic, CUDARuntimeFunctionHookerContext *);
	if (context->to_function == AttachedToFunction::RegisterFatbin) {
	}
}

static void
cuda_runtime_function_hooker_class_init(CUDARuntimeFunctionHookerClass *klass)
{
}

static void cuda_runtime_function_hooker_iface_init(gpointer g_iface,
						    gpointer iface_data)
{
	auto iface = (GumInvocationListenerInterface *)g_iface;

	iface->on_enter = example_listener_on_enter;
	iface->on_leave = example_listener_on_leave;
}

static void cuda_runtime_function_hooker_init(CUDARuntimeFunctionHooker *self)
{
}

int nv_attach_impl::detach_by_id(int id)
{
	return 0;
}

int nv_attach_impl::create_attach_with_ebpf_callback(
	ebpf_run_callback &&cb, const attach_private_data &private_data,
	int attach_type)
{
	return 0;
}
nv_attach_impl::nv_attach_impl()
{
	SPDLOG_INFO("Starting nv_attach_impl");
	gum_init_embedded();
	auto interceptor = gum_interceptor_obtain();
	assert(interceptor != nullptr);
	auto listener =
		g_object_new(cuda_runtime_function_hooker_get_type(), nullptr);
	assert(listener != nullptr);
	this->frida_interceptor = interceptor;
	this->frida_listener = listener;
	this->injector = std::make_unique<CUDAInjector>(getpid());
	gum_interceptor_begin_transaction(interceptor);
	auto ctx = std::make_unique<CUDARuntimeFunctionHookerContext>();
	ctx->to_function = AttachedToFunction::RegisterFatbin;
	ctx->impl = this;
	auto ctx_ptr = ctx.get();
	this->hooker_contexts.push_back(std::move(ctx));
	auto original___cudaRegisterFatBinary =
		(void **(*)(void *))dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
	auto result = gum_interceptor_attach(
		interceptor,
		// GSIZE_TO_POINTER(gum_module_find_export_by_name(
		// 	nullptr, "__cudaRegisterFatBinary"))
		(gpointer)original___cudaRegisterFatBinary,
		(GumInvocationListener *)listener, ctx_ptr);
	if (result != GUM_ATTACH_OK) {
		SPDLOG_ERROR("Unable to attach to CUDA functions: {}",
			     (int)result);
		assert(false);
	}
	gum_interceptor_end_transaction(interceptor);
	SPDLOG_INFO("Probe to __cudaRegisterFatBinary attached");
}

nv_attach_impl::~nv_attach_impl()
{
	if (frida_listener)
		g_object_unref(frida_listener);
}
