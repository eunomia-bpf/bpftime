#include <assert.h>
#include <sys/mount.h>
#include <sys/stat.h>

int main()
{
	/* The mount point may be missing, as in a fresh container. */
	mkdir("/mnt/ramdisk", 0755);
	int err = mount("tmpfs", "/mnt/ramdisk", "tmpfs", 0,
			(const void *)"size=1G");
	assert(err == 0);
	return 0;
}
