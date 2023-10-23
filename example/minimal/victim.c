#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int target_func() {
    printf("target_func\n");
    return 0;
}

int main(int argc, char *argv[]) {
    while(1) {
        sleep(2);
        target_func();
    }
    return 0;
}
