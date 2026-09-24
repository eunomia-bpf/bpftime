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
 * Two bank value layouts share this header and the same 335675408-byte
 * value size. The default (FRDB_SOA_LAYOUT undefined) is the measured
 * record-major AoS array of 80-byte records. With FRDB_SOA_LAYOUT
 * defined, the bank value is field-major SoA: ten field planes, each a
 * contiguous array of the bank's 16384 thread slots per record index, so
 * adjacent lanes write 8 bytes apart instead of 80.
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

#ifdef FRDB_SOA_LAYOUT
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
_Static_assert(sizeof(struct frdb_value_aos) == 335675408ULL,
	       "frdb AoS bank value layout drifted from the 320 MiB payload");
_Static_assert(sizeof(struct frdb_value_soa) == 335675408ULL,
	       "frdb SoA bank value layout drifted from the 320 MiB payload");
_Static_assert(sizeof(frdb_value) == 335675408ULL,
	       "frdb bank value layout drifted from the 320 MiB payload");

#endif
