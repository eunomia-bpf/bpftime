#include "bpftime_helper_group.hpp"
#include <cerrno>
#include <sched.h>
#ifdef ENABLE_BPFTIME_VERIFIER
#include "bpftime-verifier.hpp"
#endif
#include "spdlog/spdlog.h"
#include <map>
#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <time.h>
#include <unistd.h>
#include <ctime>
#include <filesystem>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_internal.h"
#include <spdlog/spdlog.h>
#include <vector>

#if defined (BPFTIME_ENABLE_IOURING_EXT) && __linux__
#include "liburing.h"
#endif

using namespace std;

#if BPFTIME_ENABLE_FS_HELPER
uint64_t bpftime_get_abs_path(const char *filename, const char *buffer,
			      uint64_t size)
{
	auto path = std::filesystem::absolute(filename);
	return (uint64_t)(uintptr_t)strncpy((char *)(uintptr_t)buffer,
					    path.c_str(), size);
}

uint64_t bpftime_path_join(const char *filename1, const char *filename2,
			   const char *buffer, uint64_t size)
{
	auto path = std::filesystem::path(filename1) /
		    std::filesystem::path(filename2);
	return (uint64_t)(uintptr_t)strncpy((char *)(uintptr_t)buffer,
					    path.c_str(), size);
}
#endif

namespace bpftime
{
/*
io_uring are only available in linux atm 
so adding linux guards to the following code
*/

#if defined (BPFTIME_ENABLE_IOURING_EXT) && __linux__
static int submit_io_uring_write(struct io_uring *ring, int fd, char *buf,
				 size_t size)
{
	struct io_uring_sqe *sqe;

	sqe = io_uring_get_sqe(ring);
	if (!sqe) {
		return 1;
	}
	io_uring_prep_write(sqe, fd, buf, size, -1);
	sqe->user_data = 1;

	return 0;
}

static int submit_io_uring_fsync(struct io_uring *ring, int fd)
{
	struct io_uring_sqe *sqe;

	sqe = io_uring_get_sqe(ring);
	if (!sqe) {
		return 1;
	}

	io_uring_prep_fsync(sqe, fd, IORING_FSYNC_DATASYNC);
	sqe->user_data = 2;

	return 0;
}

static int io_uring_init(struct io_uring *ring)
{
	int ret = io_uring_queue_init(1024, ring, IORING_SETUP_SINGLE_ISSUER);
	if (ret) {
		return 1;
	}
	return 0;
}

static int io_uring_wait_and_seen(struct io_uring *ring,
				  struct io_uring_cqe *cqe)
{
	int ret = io_uring_wait_cqe(ring, &cqe);
	if (ret < 0) {
		return ret;
	}
	io_uring_cqe_seen(ring, cqe);
	return 0;
}

static struct io_uring ring;

uint64_t io_uring_init_global(void)
{
	return io_uring_init(&ring);
}

uint64_t bpftime_io_uring_submit_write(int fd, char *buf, size_t size)
{
	return submit_io_uring_write(&ring, fd, buf, size);
}

uint64_t bpftime_io_uring_submit_fsync(int fd)
{
	return submit_io_uring_fsync(&ring, fd);
}

uint64_t bpftime_io_uring_wait_and_seen(void)
{
	struct io_uring_cqe *cqe = nullptr;
	return io_uring_wait_and_seen(&ring, cqe);
}

uint64_t bpftime_io_uring_submit(void)
{
	return io_uring_submit(&ring);
}
#endif

extern const bpftime_helper_group extesion_group = { {
	{ UFUNC_HELPER_ID_FIND_ID,
	  bpftime_helper_info{
		  .index = UFUNC_HELPER_ID_FIND_ID,
		  .name = "__ebpf_call_find_ufunc_id",
		  .fn = (void *)__ebpf_call_find_ufunc_id,
	  } },
	{ UFUNC_HELPER_ID_DISPATCHER,
	  bpftime_helper_info{
		  .index = UFUNC_HELPER_ID_DISPATCHER,
		  .name = "__ebpf_call_ufunc_dispatcher",
		  .fn = (void *)__ebpf_call_ufunc_dispatcher,
	  } },
#if BPFTIME_ENABLE_FS_HELPER
	{ EXTENDED_HELPER_GET_ABS_PATH_ID,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_GET_ABS_PATH_ID,
		  .name = "bpftime_get_abs_path",
		  .fn = (void *)bpftime_get_abs_path,
	  } },
	{ EXTENDED_HELPER_PATH_JOIN_ID,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_PATH_JOIN_ID,
		  .name = "bpftime_path_join",
		  .fn = (void *)bpftime_path_join,
	  } },
#endif
#if defined (BPFTIME_ENABLE_IOURING_EXT) && __linux__
	{ EXTENDED_HELPER_IOURING_INIT,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_IOURING_INIT,
		  .name = "io_uring_init",
		  .fn = (void *)io_uring_init_global,
	  } },
	{ EXTENDED_HELPER_IOURING_SUBMIT_WRITE,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_IOURING_SUBMIT_WRITE,
		  .name = "io_uring_submit_write",
		  .fn = (void *)bpftime_io_uring_submit_write,
	  } },
	{ EXTENDED_HELPER_IOURING_SUBMIT_FSYNC,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_IOURING_SUBMIT_FSYNC,
		  .name = "io_uring_submit_fsync",
		  .fn = (void *)bpftime_io_uring_submit_fsync,
	  } },
	{ EXTENDED_HELPER_IOURING_WAIT_AND_SEEN,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_IOURING_WAIT_AND_SEEN,
		  .name = "io_uring_wait_and_seen",
		  .fn = (void *)bpftime_io_uring_wait_and_seen,
	  } },
	{ EXTENDED_HELPER_IOURING_SUBMIT,
	  bpftime_helper_info{
		  .index = EXTENDED_HELPER_IOURING_SUBMIT,
		  .name = "io_uring_submit",
		  .fn = (void *)bpftime_io_uring_submit,
	  } },
#endif
} };

} // namespace bpftime
