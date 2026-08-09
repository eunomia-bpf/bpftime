#ifndef BPFTIME_EPOLL_H
#define BPFTIME_EPOLL_H
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <stddef.h>
/*
 Keeping declaration for structs that are specific to linux
 structures added from
 <sys/epoll.h>
 <linux/perf_event.h>
 <linux/types.h>
*/

#define EPOLLIN 0x001
#define BPF_OBJ_NAME_LEN 16U
#define BPF_TAG_SIZE	8
#define offsetofend(type, member) (offsetof(type, member) + sizeof(((type *)0)->member))



#ifndef __NR_bpf
#define __NR_bpf 515 // This is a placeholder value. Linux assigns syscall numbers dynamically.
#endif

#if __GNUC__ >= 4
#define LIBBPF_API __attribute__((visibility("default")))
#else
#define LIBBPF_API
#endif

#ifndef __NR_perf_event_open
#define __NR_perf_event_open 516 // This is a placeholder value. Linux assigns syscall numbers dynamically.
#endif

// Define a placeholder for PERF_EVENT_IOC_ENABLE. macOS doesn't have any equivalent for ioctl 
#define PERF_EVENT_IOC_ENABLE 0
// Define a placeholder for PERF_EVENT_IOC_DISABLE
#define PERF_EVENT_IOC_DISABLE 1 
// Define a placeholder for PERF_EVENT_IOC_SET_BPF
#define PERF_EVENT_IOC_SET_BPF 2

/* Valid opcodes to issue to sys_epoll_ctl() */ 
// link: https://github.com/torvalds/linux/blob/2ef5971ff345d3c000873725db555085e0131961/include/uapi/linux/eventpoll.h#L26
#define EPOLL_CTL_ADD 1

// https://github.com/torvalds/linux/blob/2ef5971ff345d3c000873725db555085e0131961/include/uapi/linux/bpf.h#L18
#define BPF_ALU64	0x07	/* alu mode in double word width */
#define BPF_MOV		0xb0	/* mov reg to reg */
#define		BPF_K		0x00 //https://github.com/torvalds/linux/blob/2ef5971ff345d3c000873725db555085e0131961/include/uapi/linux/bpf_common.h#L12
#define		BPF_JMP		0x05
#define BPF_EXIT	0x90	/* function return */
union epoll_data {
           void     *ptr;
           int       fd;
           uint32_t  u32;
           uint64_t  u64;
       };

typedef union epoll_data  epoll_data_t;
struct epoll_event {
           uint32_t      events;  /* Epoll events */
           epoll_data_t  data;    /* User data variable */
       };
typedef uint64_t __u64;
typedef uint64_t __aligned_u64;
typedef uint32_t __u32;
typedef uint16_t __u16;
typedef uint8_t  __u8;

typedef int16_t  __s16;
typedef int32_t  __s32;
typedef int64_t  __s64;

typedef off_t off64_t;

struct perf_event_header {
        __u32 type;
        __u16 misc;
        __u16 size;
    };

/*
https://github.com/torvalds/linux/blob/f06ce441457d4abc4d76be7acba26868a2d02b1c/include/uapi/linux/perf_event.h#L571
*/
struct perf_event_mmap_page {

	__u32	version;		/* version number of this structure */
	__u32	compat_version;		/* lowest version this is compat with */

	/*
	 * Bits needed to read the hw events in user-space.
	 *
	 *   u32 seq, time_mult, time_shift, index, width;
	 *   u64 count, enabled, running;
	 *   u64 cyc, time_offset;
	 *   s64 pmc = 0;
	 *
	 *   do {
	 *     seq = pc->lock;
	 *     barrier()
	 *
	 *     enabled = pc->time_enabled;
	 *     running = pc->time_running;
	 *
	 *     if (pc->cap_usr_time && enabled != running) {
	 *       cyc = rdtsc();
	 *       time_offset = pc->time_offset;
	 *       time_mult   = pc->time_mult;
	 *       time_shift  = pc->time_shift;
	 *     }
	 *
	 *     index = pc->index;
	 *     count = pc->offset;
	 *     if (pc->cap_user_rdpmc && index) {
	 *       width = pc->pmc_width;
	 *       pmc = rdpmc(index - 1);
	 *     }
	 *
	 *     barrier();
	 *   } while (pc->lock != seq);
	 *
	 * NOTE: for obvious reason this only works on self-monitoring
	 *       processes.
	 */
	__u32	lock;			/* seqlock for synchronization */
	__u32	index;			/* hardware event identifier */
	__s64	offset;			/* add to hardware event value */
	__u64	time_enabled;		/* time event active */
	__u64	time_running;		/* time event on cpu */
	union {
		__u64	capabilities;
		struct {
			__u64	cap_bit0		: 1, /* Always 0, deprecated, see commit 860f085b74e9 */
				cap_bit0_is_deprecated	: 1, /* Always 1, signals that bit 0 is zero */

				cap_user_rdpmc		: 1, /* The RDPMC instruction can be used to read counts */
				cap_user_time		: 1, /* The time_{shift,mult,offset} fields are used */
				cap_user_time_zero	: 1, /* The time_zero field is used */
				cap_user_time_short	: 1, /* the time_{cycle,mask} fields are used */
				cap_____res		: 58;
		};
	};

	/*
	 * If cap_user_rdpmc this field provides the bit-width of the value
	 * read using the rdpmc() or equivalent instruction. This can be used
	 * to sign extend the result like:
	 *
	 *   pmc <<= 64 - width;
	 *   pmc >>= 64 - width; // signed shift right
	 *   count += pmc;
	 */
	__u16	pmc_width;

	/*
	 * If cap_usr_time the below fields can be used to compute the time
	 * delta since time_enabled (in ns) using rdtsc or similar.
	 *
	 *   u64 quot, rem;
	 *   u64 delta;
	 *
	 *   quot = (cyc >> time_shift);
	 *   rem = cyc & (((u64)1 << time_shift) - 1);
	 *   delta = time_offset + quot * time_mult +
	 *              ((rem * time_mult) >> time_shift);
	 *
	 * Where time_offset,time_mult,time_shift and cyc are read in the
	 * seqcount loop described above. This delta can then be added to
	 * enabled and possible running (if index), improving the scaling:
	 *
	 *   enabled += delta;
	 *   if (index)
	 *     running += delta;
	 *
	 *   quot = count / running;
	 *   rem  = count % running;
	 *   count = quot * enabled + (rem * enabled) / running;
	 */
	__u16	time_shift;
	__u32	time_mult;
	__u64	time_offset;
	/*
	 * If cap_usr_time_zero, the hardware clock (e.g. TSC) can be calculated
	 * from sample timestamps.
	 *
	 *   time = timestamp - time_zero;
	 *   quot = time / time_mult;
	 *   rem  = time % time_mult;
	 *   cyc = (quot << time_shift) + (rem << time_shift) / time_mult;
	 *
	 * And vice versa:
	 *
	 *   quot = cyc >> time_shift;
	 *   rem  = cyc & (((u64)1 << time_shift) - 1);
	 *   timestamp = time_zero + quot * time_mult +
	 *               ((rem * time_mult) >> time_shift);
	 */
	__u64	time_zero;

	__u32	size;			/* Header size up to __reserved[] fields. */
	__u32	__reserved_1;

	/*
	 * If cap_usr_time_short, the hardware clock is less than 64bit wide
	 * and we must compute the 'cyc' value, as used by cap_usr_time, as:
	 *
	 *   cyc = time_cycles + ((cyc - time_cycles) & time_mask)
	 *
	 * NOTE: this form is explicitly chosen such that cap_usr_time_short
	 *       is a correction on top of cap_usr_time, and code that doesn't
	 *       know about cap_usr_time_short still works under the assumption
	 *       the counter doesn't wrap.
	 */
	__u64	time_cycles;
	__u64	time_mask;

		/*
		 * Hole for extension of the self monitor capabilities
		 */

	__u8	__reserved[116*8];	/* align to 1k. */

	/*
	 * Control data for the mmap() data buffer.
	 *
	 * User-space reading the @data_head value should issue an smp_rmb(),
	 * after reading this value.
	 *
	 * When the mapping is PROT_WRITE the @data_tail value should be
	 * written by userspace to reflect the last read data, after issueing
	 * an smp_mb() to separate the data read from the ->data_tail store.
	 * In this case the kernel will not over-write unread data.
	 *
	 * See perf_output_put_handle() for the data ordering.
	 *
	 * data_{offset,size} indicate the location and size of the perf record
	 * buffer within the mmapped area.
	 */
	__u64   data_head;		/* head in the data section */
	__u64	data_tail;		/* user-space written tail */
	__u64	data_offset;		/* where the buffer starts */
	__u64	data_size;		/* data buffer size */

	/*
	 * AUX area is defined by aux_{offset,size} fields that should be set
	 * by the userspace, so that
	 *
	 *   aux_offset >= data_offset + data_size
	 *
	 * prior to mmap()ing it. Size of the mmap()ed area should be aux_size.
	 *
	 * Ring buffer pointers aux_{head,tail} have the same semantics as
	 * data_{head,tail} and same ordering rules apply.
	 */
	__u64	aux_head;
	__u64	aux_tail;
	__u64	aux_offset;
	__u64	aux_size;
};


enum perf_event_type {

	/*
	 * If perf_event_attr.sample_id_all is set then all event types will
	 * have the sample_type selected fields related to where/when
	 * (identity) an event took place (TID, TIME, ID, STREAM_ID, CPU,
	 * IDENTIFIER) described in PERF_RECORD_SAMPLE below, it will be stashed
	 * just after the perf_event_header and the fields already present for
	 * the existing fields, i.e. at the end of the payload. That way a newer
	 * perf.data file will be supported by older perf tools, with these new
	 * optional fields being ignored.
	 *
	 * struct sample_id {
	 * 	{ u32			pid, tid; } && PERF_SAMPLE_TID
	 * 	{ u64			time;     } && PERF_SAMPLE_TIME
	 * 	{ u64			id;       } && PERF_SAMPLE_ID
	 * 	{ u64			stream_id;} && PERF_SAMPLE_STREAM_ID
	 * 	{ u32			cpu, res; } && PERF_SAMPLE_CPU
	 *	{ u64			id;	  } && PERF_SAMPLE_IDENTIFIER
	 * } && perf_event_attr::sample_id_all
	 *
	 * Note that PERF_SAMPLE_IDENTIFIER duplicates PERF_SAMPLE_ID.  The
	 * advantage of PERF_SAMPLE_IDENTIFIER is that its position is fixed
	 * relative to header.size.
	 */

	/*
	 * The MMAP events record the PROT_EXEC mappings so that we can
	 * correlate userspace IPs to code. They have the following structure:
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u32				pid, tid;
	 *	u64				addr;
	 *	u64				len;
	 *	u64				pgoff;
	 *	char				filename[];
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_MMAP			= 1,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				id;
	 *	u64				lost;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_LOST			= 2,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u32				pid, tid;
	 *	char				comm[];
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_COMM			= 3,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid, ppid;
	 *	u32				tid, ptid;
	 *	u64				time;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_EXIT			= 4,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				time;
	 *	u64				id;
	 *	u64				stream_id;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_THROTTLE			= 5,
	PERF_RECORD_UNTHROTTLE			= 6,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid, ppid;
	 *	u32				tid, ptid;
	 *	u64				time;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_FORK			= 7,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid, tid;
	 *
	 *	struct read_format		values;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_READ			= 8,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	#
	 *	# Note that PERF_SAMPLE_IDENTIFIER duplicates PERF_SAMPLE_ID.
	 *	# The advantage of PERF_SAMPLE_IDENTIFIER is that its position
	 *	# is fixed relative to header.
	 *	#
	 *
	 *	{ u64			id;	  } && PERF_SAMPLE_IDENTIFIER
	 *	{ u64			ip;	  } && PERF_SAMPLE_IP
	 *	{ u32			pid, tid; } && PERF_SAMPLE_TID
	 *	{ u64			time;     } && PERF_SAMPLE_TIME
	 *	{ u64			addr;     } && PERF_SAMPLE_ADDR
	 *	{ u64			id;	  } && PERF_SAMPLE_ID
	 *	{ u64			stream_id;} && PERF_SAMPLE_STREAM_ID
	 *	{ u32			cpu, res; } && PERF_SAMPLE_CPU
	 *	{ u64			period;   } && PERF_SAMPLE_PERIOD
	 *
	 *	{ struct read_format	values;	  } && PERF_SAMPLE_READ
	 *
	 *	{ u64			nr,
	 *	  u64			ips[nr];  } && PERF_SAMPLE_CALLCHAIN
	 *
	 *	#
	 *	# The RAW record below is opaque data wrt the ABI
	 *	#
	 *	# That is, the ABI doesn't make any promises wrt to
	 *	# the stability of its content, it may vary depending
	 *	# on event, hardware, kernel version and phase of
	 *	# the moon.
	 *	#
	 *	# In other words, PERF_SAMPLE_RAW contents are not an ABI.
	 *	#
	 *
	 *	{ u32			size;
	 *	  char                  data[size];}&& PERF_SAMPLE_RAW
	 *
	 *	{ u64                   nr;
	 *	  { u64	hw_idx; } && PERF_SAMPLE_BRANCH_HW_INDEX
	 *        { u64 from, to, flags } lbr[nr];
	 *      } && PERF_SAMPLE_BRANCH_STACK
	 *
	 * 	{ u64			abi; # enum perf_sample_regs_abi
	 * 	  u64			regs[weight(mask)]; } && PERF_SAMPLE_REGS_USER
	 *
	 * 	{ u64			size;
	 * 	  char			data[size];
	 * 	  u64			dyn_size; } && PERF_SAMPLE_STACK_USER
	 *
	 *	{ union perf_sample_weight
	 *	 {
	 *		u64		full; && PERF_SAMPLE_WEIGHT
	 *	#if defined(__LITTLE_ENDIAN_BITFIELD)
	 *		struct {
	 *			u32	var1_dw;
	 *			u16	var2_w;
	 *			u16	var3_w;
	 *		} && PERF_SAMPLE_WEIGHT_STRUCT
	 *	#elif defined(__BIG_ENDIAN_BITFIELD)
	 *		struct {
	 *			u16	var3_w;
	 *			u16	var2_w;
	 *			u32	var1_dw;
	 *		} && PERF_SAMPLE_WEIGHT_STRUCT
	 *	#endif
	 *	 }
	 *	}
	 *	{ u64			data_src; } && PERF_SAMPLE_DATA_SRC
	 *	{ u64			transaction; } && PERF_SAMPLE_TRANSACTION
	 *	{ u64			abi; # enum perf_sample_regs_abi
	 *	  u64			regs[weight(mask)]; } && PERF_SAMPLE_REGS_INTR
	 *	{ u64			phys_addr;} && PERF_SAMPLE_PHYS_ADDR
	 *	{ u64			size;
	 *	  char			data[size]; } && PERF_SAMPLE_AUX
	 *	{ u64			data_page_size;} && PERF_SAMPLE_DATA_PAGE_SIZE
	 *	{ u64			code_page_size;} && PERF_SAMPLE_CODE_PAGE_SIZE
	 * };
	 */
	PERF_RECORD_SAMPLE			= 9,

	/*
	 * The MMAP2 records are an augmented version of MMAP, they add
	 * maj, min, ino numbers to be used to uniquely identify each mapping
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u32				pid, tid;
	 *	u64				addr;
	 *	u64				len;
	 *	u64				pgoff;
	 *	union {
	 *		struct {
	 *			u32		maj;
	 *			u32		min;
	 *			u64		ino;
	 *			u64		ino_generation;
	 *		};
	 *		struct {
	 *			u8		build_id_size;
	 *			u8		__reserved_1;
	 *			u16		__reserved_2;
	 *			u8		build_id[20];
	 *		};
	 *	};
	 *	u32				prot, flags;
	 *	char				filename[];
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_MMAP2			= 10,

	/*
	 * Records that new data landed in the AUX buffer part.
	 *
	 * struct {
	 * 	struct perf_event_header	header;
	 *
	 * 	u64				aux_offset;
	 * 	u64				aux_size;
	 *	u64				flags;
	 * 	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_AUX				= 11,

	/*
	 * Indicates that instruction trace has started
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid;
	 *	u32				tid;
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_ITRACE_START		= 12,

	/*
	 * Records the dropped/lost sample number.
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *
	 *	u64				lost;
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_LOST_SAMPLES		= 13,

	/*
	 * Records a context switch in or out (flagged by
	 * PERF_RECORD_MISC_SWITCH_OUT). See also
	 * PERF_RECORD_SWITCH_CPU_WIDE.
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_SWITCH			= 14,

	/*
	 * CPU-wide version of PERF_RECORD_SWITCH with next_prev_pid and
	 * next_prev_tid that are the next (switching out) or previous
	 * (switching in) pid/tid.
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				next_prev_pid;
	 *	u32				next_prev_tid;
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_SWITCH_CPU_WIDE		= 15,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u32				pid;
	 *	u32				tid;
	 *	u64				nr_namespaces;
	 *	{ u64				dev, inode; } [nr_namespaces];
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_NAMESPACES			= 16,

	/*
	 * Record ksymbol register/unregister events:
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				addr;
	 *	u32				len;
	 *	u16				ksym_type;
	 *	u16				flags;
	 *	char				name[];
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_KSYMBOL			= 17,

	/*
	 * Record bpf events:
	 *  enum perf_bpf_event_type {
	 *	PERF_BPF_EVENT_UNKNOWN		= 0,
	 *	PERF_BPF_EVENT_PROG_LOAD	= 1,
	 *	PERF_BPF_EVENT_PROG_UNLOAD	= 2,
	 *  };
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u16				type;
	 *	u16				flags;
	 *	u32				id;
	 *	u8				tag[BPF_TAG_SIZE];
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_BPF_EVENT			= 18,

	/*
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				id;
	 *	char				path[];
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_CGROUP			= 19,

	/*
	 * Records changes to kernel text i.e. self-modified code. 'old_len' is
	 * the number of old bytes, 'new_len' is the number of new bytes. Either
	 * 'old_len' or 'new_len' may be zero to indicate, for example, the
	 * addition or removal of a trampoline. 'bytes' contains the old bytes
	 * followed immediately by the new bytes.
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				addr;
	 *	u16				old_len;
	 *	u16				new_len;
	 *	u8				bytes[];
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_TEXT_POKE			= 20,

	/*
	 * Data written to the AUX area by hardware due to aux_output, may need
	 * to be matched to the event by an architecture-specific hardware ID.
	 * This records the hardware ID, but requires sample_id to provide the
	 * event ID. e.g. Intel PT uses this record to disambiguate PEBS-via-PT
	 * records from multiple events.
	 *
	 * struct {
	 *	struct perf_event_header	header;
	 *	u64				hw_id;
	 *	struct sample_id		sample_id;
	 * };
	 */
	PERF_RECORD_AUX_OUTPUT_HW_ID		= 21,

	PERF_RECORD_MAX,			/* non-ABI */

};

struct perf_event_attr {
/*
	 * Major type: hardware/software/tracepoint/etc.
	 */
	__u32			type;

	/*
	 * Size of the attr structure, for fwd/bwd compat.
	 */
	__u32			size;

	/*
	 * Type specific configuration information.
	 */
	__u64			config;

	union {
		__u64		sample_period;
		__u64		sample_freq;
	};

	__u64			sample_type;
	__u64			read_format;

	__u64			disabled       :  1, /* off by default        */
				inherit	       :  1, /* children inherit it   */
				pinned	       :  1, /* must always be on PMU */
				exclusive      :  1, /* only group on PMU     */
				exclude_user   :  1, /* don't count user      */
				exclude_kernel :  1, /* ditto kernel          */
				exclude_hv     :  1, /* ditto hypervisor      */
				exclude_idle   :  1, /* don't count when idle */
				mmap           :  1, /* include mmap data     */
				comm	       :  1, /* include comm data     */
				freq           :  1, /* use freq, not period  */
				inherit_stat   :  1, /* per task counts       */
				enable_on_exec :  1, /* next exec enables     */
				task           :  1, /* trace fork/exit       */
				watermark      :  1, /* wakeup_watermark      */
				/*
				 * precise_ip:
				 *
				 *  0 - SAMPLE_IP can have arbitrary skid
				 *  1 - SAMPLE_IP must have constant skid
				 *  2 - SAMPLE_IP requested to have 0 skid
				 *  3 - SAMPLE_IP must have 0 skid
				 *
				 *  See also PERF_RECORD_MISC_EXACT_IP
				 */
				precise_ip     :  2, /* skid constraint       */
				mmap_data      :  1, /* non-exec mmap data    */
				sample_id_all  :  1, /* sample_type all events */

				exclude_host   :  1, /* don't count in host   */
				exclude_guest  :  1, /* don't count in guest  */

				exclude_callchain_kernel : 1, /* exclude kernel callchains */
				exclude_callchain_user   : 1, /* exclude user callchains */
				mmap2          :  1, /* include mmap with inode data     */
				comm_exec      :  1, /* flag comm events that are due to an exec */
				use_clockid    :  1, /* use @clockid for time fields */
				context_switch :  1, /* context switch data */
				write_backward :  1, /* Write ring buffer from end to beginning */
				namespaces     :  1, /* include namespaces data */
				ksymbol        :  1, /* include ksymbol events */
				bpf_event      :  1, /* include bpf events */
				aux_output     :  1, /* generate AUX records instead of events */
				cgroup         :  1, /* include cgroup events */
				text_poke      :  1, /* include text poke events */
				build_id       :  1, /* use build id in mmap2 events */
				inherit_thread :  1, /* children only inherit if cloned with CLONE_THREAD */
				remove_on_exec :  1, /* event is removed from task on exec */
				sigtrap        :  1, /* send synchronous SIGTRAP on event */
				__reserved_1   : 26;

	union {
		__u32		wakeup_events;	  /* wakeup every n events */
		__u32		wakeup_watermark; /* bytes before wakeup   */
	};

	__u32			bp_type;
	union {
		__u64		bp_addr;
		__u64		kprobe_func; /* for perf_kprobe */
		__u64		uprobe_path; /* for perf_uprobe */
		__u64		config1; /* extension of config */
	};
	union {
		__u64		bp_len;
		__u64		kprobe_addr; /* when kprobe_func == NULL */
		__u64		probe_offset; /* for perf_[k,u]probe */
		__u64		config2; /* extension of config1 */
	};
	__u64	branch_sample_type; /* enum perf_branch_sample_type */

	/*
	 * Defines set of user regs to dump on samples.
	 * See asm/perf_regs.h for details.
	 */
	__u64	sample_regs_user;

	/*
	 * Defines size of the user stack to dump on samples.
	 */
	__u32	sample_stack_user;

	__s32	clockid;
	/*
	 * Defines set of regs to dump for each sample
	 * state captured on:
	 *  - precise = 0: PMU interrupt
	 *  - precise > 0: sampled instruction
	 *
	 * See asm/perf_regs.h for details.
	 */
	__u64	sample_regs_intr;

	/*
	 * Wakeup watermark for AUX area
	 */
	__u32	aux_watermark;
	__u16	sample_max_stack;
	__u16	__reserved_2;
	__u32	aux_sample_size;
	__u32	__reserved_3;

	/*
	 * User provided data if sigtrap=1, passed back to user via
	 * siginfo_t::si_perf_data, e.g. to permit user to identify the event.
	 * Note, siginfo_t::si_perf_data is long-sized, and sig_data will be
	 * truncated accordingly on 32 bit architectures.
	 */
	__u64	sig_data;

	__u64	config3; /* extension of config2 */
};

namespace bpftime_epoll{
struct bpf_map_info {
	__u32 type;
	__u32 id;
	__u32 key_size;
	__u32 value_size;
	__u32 max_entries;
	__u32 map_flags;
	char  name[BPF_OBJ_NAME_LEN];
	__u32 ifindex;
	__u32 btf_vmlinux_value_type_id;
	__u64 netns_dev;
	__u64 netns_ino;
	__u32 btf_id;
	__u32 btf_key_type_id;
	__u32 btf_value_type_id;
	__u32 btf_vmlinux_id;
	__u64 map_extra;
} __attribute__((aligned(8)));

//https://codebrowser.dev/linux/include/linux/bpf.h.html
union bpf_attr {
	struct { /* anonymous struct used by BPF_MAP_CREATE command */
		__u32	map_type;	/* one of enum bpf_map_type */
		__u32	key_size;	/* size of key in bytes */
		__u32	value_size;	/* size of value in bytes */
		__u32	max_entries;	/* max number of entries in a map */
		__u32	map_flags;	/* BPF_MAP_CREATE related
					 * flags defined above.
					 */
		__u32	inner_map_fd;	/* fd pointing to the inner map */
		__u32	numa_node;	/* numa node (effective only if
					 * BPF_F_NUMA_NODE is set).
					 */
		char	map_name[BPF_OBJ_NAME_LEN];
		__u32	map_ifindex;	/* ifindex of netdev to create on */
		__u32	btf_fd;		/* fd pointing to a BTF type data */
		__u32	btf_key_type_id;	/* BTF type_id of the key */
		__u32	btf_value_type_id;	/* BTF type_id of the value */
		__u32	btf_vmlinux_value_type_id;/* BTF type_id of a kernel-
						   * struct stored as the
						   * map value
						   */
		__u64	map_extra;
	};
	struct { /* anonymous struct used by BPF_MAP_*_ELEM commands */
		__u32		map_fd;
		__aligned_u64	key;
		union {
			__aligned_u64 value;
			__aligned_u64 next_key;
		};
		__u64		flags;
	};
	struct { /* struct used by BPF_MAP_*_BATCH commands */
		__aligned_u64	in_batch;	/* start batch,
						 * NULL to start from beginning
						 */
		__aligned_u64	out_batch;	/* output: next start batch */
		__aligned_u64	keys;
		__aligned_u64	values;
		__u32		count;		/* input/output:
						 * input: # of key/value
						 * elements
						 * output: # of filled elements
						 */
		__u32		map_fd;
		__u64		elem_flags;
		__u64		flags;
	} batch;
	struct { /* anonymous struct used by BPF_PROG_LOAD command */
		__u32		prog_type;	/* one of enum bpf_prog_type */
		__u32		insn_cnt;
		__aligned_u64	insns;
		__aligned_u64	license;
		__u32		log_level;	/* verbosity level of verifier */
		__u32		log_size;	/* size of user buffer */
		__aligned_u64	log_buf;	/* user supplied buffer */
		__u32		kern_version;	/* not used */
		__u32		prog_flags;
		char		prog_name[BPF_OBJ_NAME_LEN];
		__u32		prog_ifindex;	/* ifindex of netdev to prep for */
		/* For some prog types expected attach type must be known at
		 * load time to verify attach type specific parts of prog
		 * (context accesses, allowed helpers, etc).
		 */
		__u32		expected_attach_type;
		__u32		prog_btf_fd;	/* fd pointing to BTF type data */
		__u32		func_info_rec_size;	/* userspace bpf_func_info size */
		__aligned_u64	func_info;	/* func info */
		__u32		func_info_cnt;	/* number of bpf_func_info records */
		__u32		line_info_rec_size;	/* userspace bpf_line_info size */
		__aligned_u64	line_info;	/* line info */
		__u32		line_info_cnt;	/* number of bpf_line_info records */
		__u32		attach_btf_id;	/* in-kernel BTF type id to attach to */
		union {
			/* valid prog_fd to attach to bpf prog */
			__u32		attach_prog_fd;
			/* or valid module BTF object fd or 0 to attach to vmlinux */
			__u32		attach_btf_obj_fd;
		};
		__u32		:32;		/* pad */
		__aligned_u64	fd_array;	/* array of FDs */
	};
	struct { /* anonymous struct used by BPF_OBJ_* commands */
		__aligned_u64	pathname;
		__u32		bpf_fd;
		__u32		file_flags;
	};
	struct { /* anonymous struct used by BPF_PROG_ATTACH/DETACH commands */
		__u32		target_fd;	/* container object to attach to */
		__u32		attach_bpf_fd;	/* eBPF program to attach */
		__u32		attach_type;
		__u32		attach_flags;
		__u32		replace_bpf_fd;	/* previously attached eBPF
						 * program to replace if
						 * BPF_F_REPLACE is used
						 */
	};
	struct { /* anonymous struct used by BPF_PROG_TEST_RUN command */
		__u32		prog_fd;
		__u32		retval;
		__u32		data_size_in;	/* input: len of data_in */
		__u32		data_size_out;	/* input/output: len of data_out
						 *   returns ENOSPC if data_out
						 *   is too small.
						 */
		__aligned_u64	data_in;
		__aligned_u64	data_out;
		__u32		repeat;
		__u32		duration;
		__u32		ctx_size_in;	/* input: len of ctx_in */
		__u32		ctx_size_out;	/* input/output: len of ctx_out
						 *   returns ENOSPC if ctx_out
						 *   is too small.
						 */
		__aligned_u64	ctx_in;
		__aligned_u64	ctx_out;
		__u32		flags;
		__u32		cpu;
	} test;
	struct { /* anonymous struct used by BPF_*_GET_*_ID */
		union {
			__u32		start_id;
			__u32		prog_id;
			__u32		map_id;
			__u32		btf_id;
			__u32		link_id;
		};
		__u32		next_id;
		__u32		open_flags;
	};
	struct { /* anonymous struct used by BPF_OBJ_GET_INFO_BY_FD */
		__u32		bpf_fd;
		__u32		info_len;
		__aligned_u64	info;
	} info;
	struct { /* anonymous struct used by BPF_PROG_QUERY command */
		__u32		target_fd;	/* container object to query */
		__u32		attach_type;
		__u32		query_flags;
		__u32		attach_flags;
		__aligned_u64	prog_ids;
		__u32		prog_cnt;
	} query;
	struct { /* anonymous struct used by BPF_RAW_TRACEPOINT_OPEN command */
		__u64 name;
		__u32 prog_fd;
	} raw_tracepoint;
	struct { /* anonymous struct for BPF_BTF_LOAD */
		__aligned_u64	btf;
		__aligned_u64	btf_log_buf;
		__u32		btf_size;
		__u32		btf_log_size;
		__u32		btf_log_level;
	};
	struct {
		__u32		pid;		/* input: pid */
		__u32		fd;		/* input: fd */
		__u32		flags;		/* input: flags */
		__u32		buf_len;	/* input/output: buf len */
		__aligned_u64	buf;		/* input/output:
						 *   tp_name for tracepoint
						 *   symbol for kprobe
						 *   filename for uprobe
						 */
		__u32		prog_id;	/* output: prod_id */
		__u32		fd_type;	/* output: BPF_FD_TYPE_* */
		__u64		probe_offset;	/* output: probe_offset */
		__u64		probe_addr;	/* output: probe_addr */
	} task_fd_query;
	struct { /* struct used by BPF_LINK_CREATE command */
		__u32		prog_fd;	/* eBPF program to attach */
		union {
			__u32		target_fd;	/* object to attach to */
			__u32		target_ifindex; /* target ifindex */
		};
		__u32		attach_type;	/* attach type */
		__u32		flags;		/* extra flags */
		union {
			__u32		target_btf_id;	/* btf_id of target to attach to */
			struct {
				__aligned_u64	iter_info;	/* extra bpf_iter_link_info */
				__u32		iter_info_len;	/* iter_info length */
			};
			struct {
				/* black box user-provided value passed through
				 * to BPF program at the execution time and
				 * accessible through bpf_get_attach_cookie() BPF helper
				 */
				__u64		bpf_cookie;
			} perf_event;
		};
	} link_create;
	struct { /* struct used by BPF_LINK_UPDATE command */
		__u32		link_fd;	/* link fd */
		/* new program fd to update link with */
		__u32		new_prog_fd;
		__u32		flags;		/* extra flags */
		/* expected link's program fd; is specified only if
		 * BPF_F_REPLACE flag is set in flags */
		__u32		old_prog_fd;
	} link_update;
	struct {
		__u32		link_fd;
	} link_detach;
	struct { /* struct used by BPF_ENABLE_STATS command */
		__u32		type;
	} enable_stats;
	struct { /* struct used by BPF_ITER_CREATE command */
		__u32		link_fd;
		__u32		flags;
	} iter_create;
	struct { /* struct used by BPF_PROG_BIND_MAP command */
		__u32		prog_fd;
		__u32		map_fd;
		__u32		flags;		/* extra flags */
	} prog_bind_map;
} __attribute__((aligned(8)));

struct bpf_insn {
	__u8	code;		/* opcode */
	__u8	dst_reg:4;	/* dest register */
	__u8	src_reg:4;	/* source register */
	__s16	off;		/* signed offset */
	__s32	imm;		/* signed immediate constant */
};

enum {
	BPF_REG_0 = 0,
	BPF_REG_1,
	BPF_REG_2,
	BPF_REG_3,
	BPF_REG_4,
	BPF_REG_5,
	BPF_REG_6,
	BPF_REG_7,
	BPF_REG_8,
	BPF_REG_9,
	BPF_REG_10,
	__MAX_BPF_REG,
};

enum bpf_cmd {
	BPF_MAP_CREATE,
	BPF_MAP_LOOKUP_ELEM,
	BPF_MAP_UPDATE_ELEM,
	BPF_MAP_DELETE_ELEM,
	BPF_MAP_GET_NEXT_KEY,
	BPF_PROG_LOAD,
	BPF_OBJ_PIN,
	BPF_OBJ_GET,
	BPF_PROG_ATTACH,
	BPF_PROG_DETACH,
	BPF_PROG_TEST_RUN,
	BPF_PROG_RUN = BPF_PROG_TEST_RUN,
	BPF_PROG_GET_NEXT_ID,
	BPF_MAP_GET_NEXT_ID,
	BPF_PROG_GET_FD_BY_ID,
	BPF_MAP_GET_FD_BY_ID,
	BPF_OBJ_GET_INFO_BY_FD,
	BPF_PROG_QUERY,
	BPF_RAW_TRACEPOINT_OPEN,
	BPF_BTF_LOAD,
	BPF_BTF_GET_FD_BY_ID,
	BPF_TASK_FD_QUERY,
	BPF_MAP_LOOKUP_AND_DELETE_ELEM,
	BPF_MAP_FREEZE,
	BPF_BTF_GET_NEXT_ID,
	BPF_MAP_LOOKUP_BATCH,
	BPF_MAP_LOOKUP_AND_DELETE_BATCH,
	BPF_MAP_UPDATE_BATCH,
	BPF_MAP_DELETE_BATCH,
	BPF_LINK_CREATE,
	BPF_LINK_UPDATE,
	BPF_LINK_GET_FD_BY_ID,
	BPF_LINK_GET_NEXT_ID,
	BPF_ENABLE_STATS,
	BPF_ITER_CREATE,
	BPF_LINK_DETACH,
	BPF_PROG_BIND_MAP,
};

enum bpf_prog_type {
	BPF_PROG_TYPE_UNSPEC,
	BPF_PROG_TYPE_SOCKET_FILTER,
	BPF_PROG_TYPE_KPROBE,
	BPF_PROG_TYPE_SCHED_CLS,
	BPF_PROG_TYPE_SCHED_ACT,
	BPF_PROG_TYPE_TRACEPOINT,
	BPF_PROG_TYPE_XDP,
	BPF_PROG_TYPE_PERF_EVENT,
	BPF_PROG_TYPE_CGROUP_SKB,
	BPF_PROG_TYPE_CGROUP_SOCK,
	BPF_PROG_TYPE_LWT_IN,
	BPF_PROG_TYPE_LWT_OUT,
	BPF_PROG_TYPE_LWT_XMIT,
	BPF_PROG_TYPE_SOCK_OPS,
	BPF_PROG_TYPE_SK_SKB,
	BPF_PROG_TYPE_CGROUP_DEVICE,
	BPF_PROG_TYPE_SK_MSG,
	BPF_PROG_TYPE_RAW_TRACEPOINT,
	BPF_PROG_TYPE_CGROUP_SOCK_ADDR,
	BPF_PROG_TYPE_LWT_SEG6LOCAL,
	BPF_PROG_TYPE_LIRC_MODE2,
	BPF_PROG_TYPE_SK_REUSEPORT,
	BPF_PROG_TYPE_FLOW_DISSECTOR,
	BPF_PROG_TYPE_CGROUP_SYSCTL,
	BPF_PROG_TYPE_RAW_TRACEPOINT_WRITABLE,
	BPF_PROG_TYPE_CGROUP_SOCKOPT,
	BPF_PROG_TYPE_TRACING,
	BPF_PROG_TYPE_STRUCT_OPS,
	BPF_PROG_TYPE_EXT,
	BPF_PROG_TYPE_LSM,
	BPF_PROG_TYPE_SK_LOOKUP,
	BPF_PROG_TYPE_SYSCALL, /* a program that can execute syscalls */
	BPF_PROG_TYPE_NETFILTER,
	__MAX_BPF_PROG_TYPE
};

enum bpf_attach_type {
	BPF_CGROUP_INET_INGRESS,
	BPF_CGROUP_INET_EGRESS,
	BPF_CGROUP_INET_SOCK_CREATE,
	BPF_CGROUP_SOCK_OPS,
	BPF_SK_SKB_STREAM_PARSER,
	BPF_SK_SKB_STREAM_VERDICT,
	BPF_CGROUP_DEVICE,
	BPF_SK_MSG_VERDICT,
	BPF_CGROUP_INET4_BIND,
	BPF_CGROUP_INET6_BIND,
	BPF_CGROUP_INET4_CONNECT,
	BPF_CGROUP_INET6_CONNECT,
	BPF_CGROUP_INET4_POST_BIND,
	BPF_CGROUP_INET6_POST_BIND,
	BPF_CGROUP_UDP4_SENDMSG,
	BPF_CGROUP_UDP6_SENDMSG,
	BPF_LIRC_MODE2,
	BPF_FLOW_DISSECTOR,
	BPF_CGROUP_SYSCTL,
	BPF_CGROUP_UDP4_RECVMSG,
	BPF_CGROUP_UDP6_RECVMSG,
	BPF_CGROUP_GETSOCKOPT,
	BPF_CGROUP_SETSOCKOPT,
	BPF_TRACE_RAW_TP,
	BPF_TRACE_FENTRY,
	BPF_TRACE_FEXIT,
	BPF_MODIFY_RETURN,
	BPF_LSM_MAC,
	BPF_TRACE_ITER,
	BPF_CGROUP_INET4_GETPEERNAME,
	BPF_CGROUP_INET6_GETPEERNAME,
	BPF_CGROUP_INET4_GETSOCKNAME,
	BPF_CGROUP_INET6_GETSOCKNAME,
	BPF_XDP_DEVMAP,
	BPF_CGROUP_INET_SOCK_RELEASE,
	BPF_XDP_CPUMAP,
	BPF_SK_LOOKUP,
	BPF_XDP,
	BPF_SK_SKB_VERDICT,
	BPF_SK_REUSEPORT_SELECT,
	BPF_SK_REUSEPORT_SELECT_OR_MIGRATE,
	BPF_PERF_EVENT,
	BPF_TRACE_KPROBE_MULTI,
	BPF_LSM_CGROUP,
	BPF_STRUCT_OPS,
	BPF_NETFILTER,
	BPF_TCX_INGRESS,
	BPF_TCX_EGRESS,
	BPF_TRACE_UPROBE_MULTI,
	BPF_CGROUP_UNIX_CONNECT,
	BPF_CGROUP_UNIX_SENDMSG,
	BPF_CGROUP_UNIX_RECVMSG,
	BPF_CGROUP_UNIX_GETPEERNAME,
	BPF_CGROUP_UNIX_GETSOCKNAME,
	BPF_NETKIT_PRIMARY,
	BPF_NETKIT_PEER,
	BPF_TRACE_KPROBE_SESSION,
	__MAX_BPF_ATTACH_TYPE
};

/* Short form of mov, dst_reg = imm32 */

#define BPF_MOV64_IMM(DST, IMM)					\
	((struct bpf_insn) {					\
		.code  = BPF_ALU64 | BPF_MOV | BPF_K,		\
		.dst_reg = DST,					\
		.src_reg = 0,					\
		.off   = 0,					\
		.imm   = IMM })

#define BPF_EXIT_INSN()						\
	((struct bpf_insn) {					\
		.code  = BPF_JMP | BPF_EXIT,			\
		.dst_reg = 0,					\
		.src_reg = 0,					\
		.off   = 0,					\
		.imm   = 0 })

struct bpf_prog_info {
	__u32 type;
	__u32 id;
	__u8  tag[BPF_TAG_SIZE];
	__u32 jited_prog_len;
	__u32 xlated_prog_len;
	__aligned_u64 jited_prog_insns;
	__aligned_u64 xlated_prog_insns;
	__u64 load_time;	/* ns since boottime */
	__u32 created_by_uid;
	__u32 nr_map_ids;
	__aligned_u64 map_ids;
	char name[BPF_OBJ_NAME_LEN];
	__u32 ifindex;
	__u32 gpl_compatible:1;
	__u32 :31; /* alignment pad */
	__u64 netns_dev;
	__u64 netns_ino;
	__u32 nr_jited_ksyms;
	__u32 nr_jited_func_lens;
	__aligned_u64 jited_ksyms;
	__aligned_u64 jited_func_lens;
	__u32 btf_id;
	__u32 func_info_rec_size;
	__aligned_u64 func_info;
	__u32 nr_func_info;
	__u32 nr_line_info;
	__aligned_u64 line_info;
	__aligned_u64 jited_line_info;
	__u32 nr_jited_line_info;
	__u32 line_info_rec_size;
	__u32 jited_line_info_rec_size;
	__u32 nr_prog_tags;
	__aligned_u64 prog_tags;
	__u64 run_time_ns;
	__u64 run_cnt;
	__u64 recursion_misses;
	__u32 verified_insns;
	__u32 attach_btf_obj_id;
	__u32 attach_btf_id;
} __attribute__((aligned(8)));

static inline __u64 ptr_to_u64(const void *ptr)
{
	return (__u64) (unsigned long) ptr;
}
static inline int sys_bpf(enum bpf_cmd cmd, union bpf_attr *attr,
			  unsigned int size)
{
	// No-op stub
	return 0;
}

// from libbpf/btf
void btf__free(struct btf *btf);
// from libbpf
void
bpf_object__close(struct bpf_object* object); 

LIBBPF_API struct bpf_program *
bpf_object__next_program(const struct bpf_object *obj, struct bpf_program *prog);

#define bpf_object__for_each_program(pos, obj)			\
	for ((pos) = bpf_object__next_program((obj), NULL);	\
	     (pos) != NULL;					\
	     (pos) = bpf_object__next_program((obj), (pos)))

struct bpf_program;

LIBBPF_API const struct bpf_insn *bpf_program__insns(const struct bpf_program *prog);
LIBBPF_API size_t bpf_program__insn_cnt(const struct bpf_program *prog);
LIBBPF_API const char *bpf_program__name(const struct bpf_program *prog);
LIBBPF_API const char *bpf_program__section_name(const struct bpf_program *prog);
LIBBPF_API struct bpf_object *bpf_object__open(const char *path);

enum libbpf_print_level {
        LIBBPF_WARN,
        LIBBPF_INFO,
        LIBBPF_DEBUG,
};

// https://github.com/torvalds/linux/blob/2ef5971ff345d3c000873725db555085e0131961/tools/lib/bpf/bpf.h#L503
inline int bpf_obj_get_info_by_fd(int bpf_fd, void *info, __u32 *info_len)
{
	const size_t attr_sz = offsetofend(union bpf_attr, info);
	union bpf_attr attr;
	int err;

	memset(&attr, 0, attr_sz);
	attr.info.bpf_fd = bpf_fd;
	attr.info.info_len = *info_len;
	attr.info.info = ptr_to_u64(info);

	err = sys_bpf(BPF_OBJ_GET_INFO_BY_FD, &attr, attr_sz);
	if (!err)
		*info_len = attr.info.info_len;
	// no-op stub
	// return libbpf_err_errno(err);
	return 1;
}
}
#endif