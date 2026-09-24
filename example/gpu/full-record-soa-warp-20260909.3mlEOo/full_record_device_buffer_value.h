/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 *
 * GPU-local full-record device buffer layout, shared byte-for-byte by the
 * BPF program and the host collector. The BPF side defines
 * FULL_RECORD_VALUE_U64 (to u64 from vmlinux.h) before including this
 * header; the host side falls back to uint64_t.
 *
 * The full per-thread record set (524288 slots x 256 records x 80 bytes,
 * about 10 GiB) is split into 32 equal banks of 16384 thread slots.
 * Each value stays below 512 MiB: the first 1.25-GiB bank was truncated
 * in the actual compiler BTF output despite its correct C sizeof.
 *
 * Three bank value layouts share this header and the same 335675408-byte
 * value size. The default (no layout flag) is the measured record-major
 * AoS array of 80-byte records. With FRDB_SOA_LAYOUT defined, the bank
 * value is field-major SoA: ten field planes, each a contiguous array of
 * the bank's 16384 thread slots per record index, so adjacent lanes write
 * 8 bytes apart instead of 80. With FRDB_AOSOA_LAYOUT defined, the bank
 * value is 32-slot grouped SoA (AoSoA): within each record index, every
 * group of 32 consecutive bank slots keeps its ten fields in 2560
 * contiguous bytes, so the 32 lanes of a group write 8 bytes apart within
 * each field while one slot's ten fields stay inside that 2560-byte
 * window instead of ten widely separated planes.
 *
 * FRDB_WARP_CONTIGUOUS_INDEX (only meaningful together with
 * FRDB_SOA_LAYOUT) does not change the bank value at all: it only
 * selects the BPF writer's storage slot bijection, the standard
 * block-major/thread-major order instead of the default row-major one,
 * so adjacent physical warp lanes map to adjacent slots for the
 * measured pp512 rope geometry. The host collector is unchanged.
 */
#ifndef FULL_RECORD_DEVICE_BUFFER_VALUE_H
#define FULL_RECORD_DEVICE_BUFFER_VALUE_H

#ifdef FULL_RECORD_VALUE_U64
typedef FULL_RECORD_VALUE_U64 frdb_u64;
#else
#include <stdint.h>
typedef uint64_t frdb_u64;
#endif

#define FRDB_NUM_BANKS 32ULL
#define FRDB_SLOTS_PER_BANK 16384ULL
#define FRDB_TOTAL_SLOTS (FRDB_NUM_BANKS * FRDB_SLOTS_PER_BANK) /* 524288 */
#define FRDB_RECORDS_PER_SLOT 256ULL
/* One SoA field plane: all record indices x all thread slots of a bank. */
#define FRDB_PLANE_ELEMS (FRDB_RECORDS_PER_SLOT * FRDB_SLOTS_PER_BANK)
/* Grouped SoA (AoSoA): 32 consecutive bank slots per group; ten fields
 * per record. */
#define FRDB_GROUP_SLOTS 32ULL
#define FRDB_GROUPS_PER_BANK (FRDB_SLOTS_PER_BANK / FRDB_GROUP_SLOTS) /* 512 */
#define FRDB_NUM_FIELDS 10ULL

/* One original kernelretsnoop record: all ten u64 fields, 80 bytes. */
struct frdb_record {
	frdb_u64 block_x, block_y, block_z;
	frdb_u64 thread_x, thread_y, thread_z;
	frdb_u64 block_dim_x, block_dim_y, block_dim_z;
	frdb_u64 timestamp;
};

/* Default measured layout: record-major AoS. */
struct frdb_value_aos {
	/* Bank-local: appends rejected because a slot's 256-record capacity
	 * was reached. */
	frdb_u64 total_overflow;
	/* Global sink: threads whose per-thread linear coordinate exceeds the
	 * 524288-slot capacity are counted in bank 0, because an out-of-range
	 * coordinate maps to no bank. */
	frdb_u64 total_out_of_range;
	/* One 64-bit append counter per thread slot in this bank. */
	frdb_u64 slot_counters[FRDB_SLOTS_PER_BANK];
	/* Record-major storage: for a fixed record index (append counter),
	 * all slots in this bank are contiguous, so simultaneous
	 * appends by adjacent threads land 80 bytes apart instead of
	 * ~20 KiB apart. A slot's record k sits at
	 * records[k * FRDB_SLOTS_PER_BANK + slot]. */
	struct frdb_record
		records[FRDB_SLOTS_PER_BANK * FRDB_RECORDS_PER_SLOT];
};

/* Opt-in layout (FRDB_SOA_LAYOUT): field-major SoA. Within each record
 * index k, the ten fields are ten contiguous planes of 16384 u64 thread
 * slots; the slot's field value at record k sits at
 * plane[k * FRDB_SLOTS_PER_BANK + slot], so simultaneous appends by
 * adjacent lanes land 8 bytes apart. The counters, per-slot capacity, and
 * value size are identical to the AoS layout. */
struct frdb_value_soa {
	frdb_u64 total_overflow;
	frdb_u64 total_out_of_range;
	frdb_u64 slot_counters[FRDB_SLOTS_PER_BANK];
	frdb_u64 block_x[FRDB_PLANE_ELEMS];
	frdb_u64 block_y[FRDB_PLANE_ELEMS];
	frdb_u64 block_z[FRDB_PLANE_ELEMS];
	frdb_u64 thread_x[FRDB_PLANE_ELEMS];
	frdb_u64 thread_y[FRDB_PLANE_ELEMS];
	frdb_u64 thread_z[FRDB_PLANE_ELEMS];
	frdb_u64 block_dim_x[FRDB_PLANE_ELEMS];
	frdb_u64 block_dim_y[FRDB_PLANE_ELEMS];
	frdb_u64 block_dim_z[FRDB_PLANE_ELEMS];
	frdb_u64 timestamp[FRDB_PLANE_ELEMS];
};

/* Opt-in layout (FRDB_AOSOA_LAYOUT): 32-slot grouped SoA (AoSoA). For a
 * bank-local slot s, record index k, and field f in [0, 10), the u64 sits
 * at fields[k][s / 32][f][s % 32], i.e. u64 index
 * (((k * (16384 / 32) + s / 32) * 10 + f) * 32 + s % 32). All ten fields
 * of the 32 slots of one group occupy 2560 contiguous bytes at a fixed
 * k, so the 32 lanes of a group write 8 bytes apart within each field
 * while a slot's ten fields stay inside that 2560-byte window. The
 * counters, per-slot capacity, and value size are identical to the AoS
 * and SoA layouts. */
struct frdb_value_aosoa {
	frdb_u64 total_overflow;
	frdb_u64 total_out_of_range;
	frdb_u64 slot_counters[FRDB_SLOTS_PER_BANK];
	frdb_u64 fields[FRDB_RECORDS_PER_SLOT][FRDB_GROUPS_PER_BANK]
		   [FRDB_NUM_FIELDS][FRDB_GROUP_SLOTS];
};

#if defined(FRDB_AOSOA_LAYOUT) && defined(FRDB_SOA_LAYOUT)
#error FRDB_AOSOA_LAYOUT and FRDB_SOA_LAYOUT are mutually exclusive
#endif

#if defined(FRDB_WARP_CONTIGUOUS_INDEX) && !defined(FRDB_SOA_LAYOUT)
#error FRDB_WARP_CONTIGUOUS_INDEX is defined only with FRDB_SOA_LAYOUT
#endif

#if defined(FRDB_AOSOA_LAYOUT)
typedef struct frdb_value_aosoa frdb_value;
#define FRDB_VALUE_TYPE struct frdb_value_aosoa
#define FRDB_LAYOUT_LABEL "32-slot grouped SoA (AoSoA)"
#elif defined(FRDB_SOA_LAYOUT) && defined(FRDB_WARP_CONTIGUOUS_INDEX)
typedef struct frdb_value_soa frdb_value;
#define FRDB_VALUE_TYPE struct frdb_value_soa
#define FRDB_LAYOUT_LABEL "field-major SoA (warp-contiguous index)"
#elif defined(FRDB_SOA_LAYOUT)
typedef struct frdb_value_soa frdb_value;
#define FRDB_VALUE_TYPE struct frdb_value_soa
#define FRDB_LAYOUT_LABEL "field-major SoA"
#else
typedef struct frdb_value_aos frdb_value;
#define FRDB_VALUE_TYPE struct frdb_value_aos
#define FRDB_LAYOUT_LABEL "record-major AoS"
#endif

_Static_assert(sizeof(struct frdb_record) == 80,
	       "frdb record must stay 80 bytes");
_Static_assert(FRDB_NUM_FIELDS * 8 == sizeof(struct frdb_record),
	       "frdb field count must match the 80-byte record");
_Static_assert(sizeof(struct frdb_value_aos) == 335675408ULL,
	       "frdb AoS bank value layout drifted from the 320 MiB payload");
_Static_assert(sizeof(struct frdb_value_soa) == 335675408ULL,
	       "frdb SoA bank value layout drifted from the 320 MiB payload");
_Static_assert(sizeof(struct frdb_value_aosoa) == 335675408ULL,
	       "frdb AoSoA bank value layout drifted from the 320 MiB payload");
_Static_assert(sizeof(frdb_value) == 335675408ULL,
	       "frdb bank value layout drifted from the 320 MiB payload");

#endif
