#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    putchar('i');
    printf("\nHello malloc!\n");
    for (int i = 0; i < 10; i++) {
        void *p = malloc(1024);
        printf("continue malloc...\n");
        sleep(1);
        free(p);
    }
    return 0;
}