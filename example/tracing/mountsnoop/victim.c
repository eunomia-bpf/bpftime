#include <assert.h>
#include <errno.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/types.h>

int main()
{
	/* mount(2) requires the target directory to exist. Container and CI
	 * images do not always pre-create /mnt/ramdisk, in which case the
	 * mount fails with ENOENT and the assert below aborts the victim.
	 * Create the mount point here so the example is self-contained;
	 * tolerate EEXIST so repeated runs against an existing dir work. */
	if (mkdir("/mnt/ramdisk", 0755) != 0 && errno != EEXIST)
		return 1;
	int err = mount("tmpfs", "/mnt/ramdisk", "tmpfs", 0,
			(const void *)"size=1G");
	assert(err == 0);
	return 0;
}
