#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int target_func()
{
	int res = open("/dev/null");
	printf("target_func is running, open res = %d\n", res);
	if (res > 0) {
		close(res);
	}
	return 0;
}

int main(int argc, char *argv[])
{
	while (1) {
		sleep(1);
		int res = target_func();
		if (res != 0) {
			printf("got error %d\n", res);
		}
	}
	return 0;
}
