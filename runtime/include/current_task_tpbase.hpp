/* SPDX-License-Identifier: MIT
 *
 * Resolver for the thread-pointer base slot of the synthetic per-thread
 * task_struct, shared by the in-process BPF runtime and the rootless
 * syscall server.
 *
 * On a normal host the offset of task_struct.thread.fsbase (x86-64) or
 * task_struct.thread.uw.tp_value (arm64) is read from kernel BTF. Some BTF
 * blobs (observed on 5.4-based kernels) carry several BTF_KIND_STRUCT
 * entries named "task_struct", including empty forward declarations, so a
 * plain name lookup either fails with "multiple matching types" or resolves
 * to the empty entry. Every non-empty candidate is scanned here and the
 * first one carrying the full member path wins.
 *
 * When BTF is absent, unreadable or does not expose a sane offset, a fixed
 * fallback is used. The synthetic task is owned by bpftime, so the slot only
 * has to stay within the sanity window of the userspace binary-analysis
 * decoders that consume it.
 */
#ifndef BPFTIME_CURRENT_TASK_TPBASE_HPP
#define BPFTIME_CURRENT_TASK_TPBASE_HPP

#include <cstdint>

#if defined(__linux__) && \
	(defined(__x86_64__) || defined(__aarch64__)) && \
	defined(BPFTIME_BUILD_WITH_LIBBPF)
#include <bpf/btf.h>
#include <bpf/libbpf.h>
#include <linux/btf.h>
#include <cstring>
#endif

namespace bpftime
{

// Used when kernel BTF cannot answer the query. 8-byte aligned for the
// arm64 LDR encoding and inside the [500, 20000] sanity window applied by
// generic binary-analysis decoders on both architectures.
constexpr uint32_t FALLBACK_CURRENT_TASK_TPBASE_OFFSET = 2048;

// Encode `ldr x0, [x0, #offset]` (LDR, unsigned immediate, 64-bit), the
// instruction tls_set uses to load thread.uw.tp_value and the pattern the
// arm64 binary-analysis decoder recognizes. Offset must be 8-byte aligned
// and smaller than 64 KiB; the resolver's sanity check guarantees that.
constexpr uint32_t encode_arm64_tls_set_insn(uint32_t offset)
{
	return 0xf9400000U | (((offset / 8U) & 0xfffU) << 10);
}

#if defined(__linux__) && \
	(defined(__x86_64__) || defined(__aarch64__)) && \
	defined(BPFTIME_BUILD_WITH_LIBBPF)
// Byte offset of member `name` in struct/union `type_id`, looking through
// anonymous struct/union members. On success *member_type receives the
// resolved type id of the matched member.
inline long btf_task_member_offset(const struct btf *btf, int type_id,
				   const char *name, int *member_type)
{
	const struct btf_type *type = btf__type_by_id(btf, type_id);
	if (type == nullptr ||
	    !(btf_is_struct(type) || btf_is_union(type)))
		return -1;
	const struct btf_member *member = btf_members(type);
	for (int i = 0; i < btf_vlen(type); i++, member++) {
		long offset = btf_member_bit_offset(type, i) / 8;
		int resolved = btf__resolve_type(btf, member->type);
		const char *member_name =
			btf__name_by_offset(btf, member->name_off);
		if (member_name != nullptr && member_name[0] != '\0') {
			if (strcmp(member_name, name) != 0)
				continue;
			*member_type = resolved;
			return offset;
		}
		long inner =
			btf_task_member_offset(btf, resolved, name,
					       member_type);
		if (inner >= 0)
			return offset + inner;
	}
	return -1;
}

inline bool tpbase_offset_is_sane(uint32_t offset)
{
#if defined(__x86_64__)
	// Matches the userspace decoder's accepted task_struct slot range.
	return offset >= 500 && offset <= 20000;
#else
	// arm64 LDR requires an 8-byte aligned offset and the decoder rejects
	// values outside [64, 65536).
	return offset >= 64 && offset < 65536 && (offset % 8U == 0);
#endif
}

// Returns the task_struct thread-pointer byte offset, or the fallback.
// Cached after the first parse: BTF is constant for the lifetime of the
// host and the result feeds every system-analysis completion.
inline uint32_t resolve_current_task_tpbase_offset() noexcept
{
	static const uint32_t cached_offset = []() -> uint32_t {
		struct btf *btf = btf__parse_raw("/sys/kernel/btf/vmlinux");
		if (btf == nullptr || libbpf_get_error(btf) != 0)
			return FALLBACK_CURRENT_TASK_TPBASE_OFFSET;
#if defined(__x86_64__)
		const char *member_path[] = { "thread", "fsbase" };
#else
		const char *member_path[] = { "thread", "uw", "tp_value" };
#endif
		uint32_t result = FALLBACK_CURRENT_TASK_TPBASE_OFFSET;
		__u32 type_count = btf__type_cnt(btf);
		for (__u32 id = 1; id < type_count; id++) {
			const struct btf_type *type =
				btf__type_by_id(btf, id);
			if (type == nullptr || !btf_is_struct(type) ||
			    btf_vlen(type) == 0)
				continue;
			const char *name =
				btf__name_by_offset(btf, type->name_off);
			if (name == nullptr ||
			    strcmp(name, "task_struct") != 0)
				continue;
			int current_id = (int)id;
			long total = 0;
			bool ok = true;
			for (const char *member_name : member_path) {
				int next_id = 0;
				long member = btf_task_member_offset(
					btf, current_id, member_name,
					&next_id);
				if (member < 0) {
					ok = false;
					break;
				}
				total += member;
				current_id = next_id;
			}
			if (ok && total > 0 &&
			    tpbase_offset_is_sane((uint32_t)total)) {
				result = (uint32_t)total;
				break;
			}
		}
		btf__free(btf);
		return result;
	}();
	return cached_offset;
}
#else
inline uint32_t resolve_current_task_tpbase_offset() noexcept
{
	return FALLBACK_CURRENT_TASK_TPBASE_OFFSET;
}
#endif

} // namespace bpftime

#endif // BPFTIME_CURRENT_TASK_TPBASE_HPP
