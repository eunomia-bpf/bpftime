#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <unistd.h>

__device__ __noinline__ unsigned long long bb_reg_odd_path(unsigned int marker)
{
	unsigned long long out = 0;
	asm volatile("{\n\t"
		     ".reg .b64 %%rd<3>;\n\t"
		     "cvt.u64.u32 %%rd1, %1;\n\t"
		     "xor.b64 %%rd2, %%rd1, 4660;\n\t"
		     "mov.u64 %0, %%rd2;\n\t"
		     "}\n\t"
		     : "=l"(out)
		     : "r"(marker));
	return out;
}

extern "C" __global__ void bb_reg_kernel(unsigned long long *out)
{
	asm volatile(".reg .b64 %%bbreg_out_ptr;\n\t"
		     "mov.u64 %%bbreg_out_ptr, %0;\n\t"
		     "{\n\t"
		     ".reg .pred %%p<5>;\n\t"
		     ".reg .b32 %%r<5>;\n\t"
		     ".reg .b64 %%rd<9>;\n\t"
		     "mov.u64 %%rd4, %%bbreg_out_ptr;\n\t"
		     "mov.u32 %%r1, %%tid.x;\n\t"
		     "xor.b32 %%r2, %%r1, -559038737;\n\t"
		     "and.b32 %%r3, %%r1, 1;\n\t"
		     "setp.eq.b32 %%p1, %%r3, 1;\n\t"
		     "mov.pred %%p2, 0;\n\t"
		     "xor.pred %%p3, %%p1, %%p2;\n\t"
		     "not.pred %%p4, %%p3;\n\t"
		     "@%%p4 bra $L__BB1_2;\n\t"
		     "bra.uni $L__BB1_1;\n\t"
		     "$L__BB1_2:\n\t"
		     "add.s32 %%r4, %%r2, 1;\n\t"
		     "cvt.u64.u32 %%rd8, %%r4;\n\t"
		     "bra.uni $L__BB1_3;\n\t"
		     "$L__BB1_1:\n\t"
		     "cvt.u64.u32 %%rd1, %%r2;\n\t"
		     "xor.b64 %%rd8, %%rd1, 4660;\n\t"
		     "$L__BB1_3:\n\t"
		     "cvta.to.global.u64 %%rd5, %%rd4;\n\t"
		     "mul.wide.u32 %%rd6, %%r1, 8;\n\t"
		     "add.s64 %%rd7, %%rd5, %%rd6;\n\t"
		     "st.global.u64 [%%rd7], %%rd8;\n\t"
		     "}\n\t"
		     :
		     : "l"(out)
		     : "memory");
}

int main()
{
	constexpr int kThreads = 1;
	unsigned long long *d_out = nullptr;
	unsigned long long h_out[kThreads] = {};

	cudaError_t err = cudaMalloc(&d_out, sizeof(h_out));
	if (err != cudaSuccess) {
		std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	for (int iter = 0;; iter++) {
		bb_reg_kernel<<<1, kThreads>>>(d_out);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
				     cudaGetErrorString(err));
			break;
		}

		err = cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::fprintf(stderr, "cudaMemcpy failed: %s\n",
				     cudaGetErrorString(err));
			break;
		}

		std::printf("iter=%d lane0=0x%llx expected_lane0=0xdeadbef0\n", iter,
			    h_out[0]);
		std::fflush(stdout);
		sleep(1);
	}

	cudaFree(d_out);
	return 0;
}
