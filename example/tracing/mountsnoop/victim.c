#include <assert.h>
#include <sys/mount.h>

int main()
{
	int err = mount("tmpfs", "/mnt/ramdisk", "tmpfs", 0,
			(const void *)"size=1G");
	assert(err == 0);
	return 0;
}
