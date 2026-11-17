#include <assert.h>
#include <errno.h>
#include <sys/mount.h>
#include <sys/stat.h>

int main()
{
	/* mount(2) needs an existing mountpoint: without it the call fails
	 * with -ENOENT before the filesystem is reached, and the traced
	 * line no longer shows a successful mount. EEXIST is fine, an
	 * earlier run may have left the directory behind. */
	if (mkdir("/mnt/ramdisk", 0755) != 0)
		assert(errno == EEXIST);
	int err = mount("tmpfs", "/mnt/ramdisk", "tmpfs", 0,
			(const void *)"size=1G");
	assert(err == 0);
	return 0;
}
