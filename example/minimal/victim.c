#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int target_func() {
    int res = open("/dev/null");
    printf("target_func\n");
    close(res);
    return 0;
}

int main(int argc, char *argv[]) {
    while(1) {
        sleep(1);
        target_func();
    }
    return 0;
}
