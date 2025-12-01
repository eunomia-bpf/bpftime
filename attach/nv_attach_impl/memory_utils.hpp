#ifndef _MEMORY_UTILS_HPP
#define _MEMORY_UTILS_HPP

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/uio.h>
namespace bpftime
{
namespace attach
{
// You would replace this with your own memory reading utility.
namespace memory_utils
{
static inline ssize_t process_vm_readv(pid_t pid, const struct iovec *local_iov,
				       unsigned long liovcnt,
				       const struct iovec *remote_iov,
				       unsigned long riovcnt,
				       unsigned long flags)
{
	return syscall(SYS_process_vm_readv, pid, local_iov, liovcnt,
		       remote_iov, riovcnt, flags);
}
template <typename T>
bool read_memory(pid_t pid, const void *remote_addr, T *out_value)
{
	// 首先尝试使用 process_vm_readv
	struct iovec local_iov = { .iov_base = out_value,
				   .iov_len = sizeof(T) };

	struct iovec remote_iov = { .iov_base = const_cast<void *>(remote_addr),
				    .iov_len = sizeof(T) };

	ssize_t read = memory_utils::process_vm_readv(pid, &local_iov, 1,
						      &remote_iov, 1, 0);
	if (read == sizeof(T)) {
		return true;
	}

	// 如果 process_vm_readv 失败，尝试使用 ptrace
	// 注意：这种方法需要进程被暂停（通过 PTRACE_ATTACH 或其他方式）

	// 对于不同大小的数据类型，我们可能需要多次读取
	const size_t word_size = sizeof(long);
	const size_t num_words = (sizeof(T) + word_size - 1) / word_size;

	uint8_t *buffer = reinterpret_cast<uint8_t *>(out_value);
	uintptr_t addr = reinterpret_cast<uintptr_t>(remote_addr);

	for (size_t i = 0; i < num_words; ++i) {
		errno = 0;
		long word = ptrace(PTRACE_PEEKDATA, pid, addr + (i * word_size),
				   nullptr);

		if (errno != 0) {
			return false;
		}

		// 计算这个字应该复制多少字节
		size_t bytes_to_copy =
			std::min(word_size, sizeof(T) - (i * word_size));

		// 复制数据到输出缓冲区
		std::memcpy(buffer + (i * word_size), &word, bytes_to_copy);
	}

	return true;
}
} // namespace memory_utils

} // namespace attach
} // namespace bpftime

#endif
