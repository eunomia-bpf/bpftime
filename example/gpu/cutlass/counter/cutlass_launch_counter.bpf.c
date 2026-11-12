#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

// GPU helper functions (bpftime CUDA attach)
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} cutlass_call_count SEC(".maps");

SEC("kretprobe/_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1ELb0ENSD_9NoPermuteEEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1ELb0ESJ_EENSL_ISQ_fSE_Li0ESS_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSM_fSE_fSE_NSW_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS15_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2ENS8_6thread14UnaryTransform8IdentityEEES1F_bEENS_8epilogue11threadblock8EpilogueIS7_S16_Li1ENS1I_22PredicatedTileIteratorINS1I_26OutputTileOptimalThreadMapINS1I_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1M_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESJ_Lb0EEENS1H_4warp20FragmentIteratorSimtISY_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSM_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S14_EENS1R_16TileIteratorSimtISY_S1Y_fSE_S14_EENS1I_18SharedLoadIteratorINS1P_18CompactedThreadMapEfLi4EEENS1H_6thread17LinearCombinationIfLi1EffLNS25_9ScaleType4KindE0ELS1B_2EfEENSB_ILi0ELi17EEELi1ELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")
int cuda__cutlass_launch(void *ctx)
{
	u32 key = 0;
	// Count once per kernel launch to avoid massive contention and to ensure
	// host visibility via map update (system-scope fence in trampoline).
	u64 tx = 0, ty = 0, tz = 0;
	u64 bx = 0, by = 0, bz = 0;
	bpf_get_thread_idx(&tx, &ty, &tz);
	bpf_get_block_idx(&bx, &by, &bz);
	if (tx == 0 && ty == 0 && tz == 0 && bx == 0 && by == 0 && bz == 0) {
		u64 *cnt = bpf_map_lookup_elem(&cutlass_call_count, &key);
		u64 newv = 1;
		if (cnt)
			newv = *cnt + 1;
		bpf_map_update_elem(&cutlass_call_count, &key, &newv,
				    (u64)BPF_ANY);
	}
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
