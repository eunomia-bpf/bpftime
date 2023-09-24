#include <iostream>
#include <unistd.h>
#include <limits.h>

int main() {
    char execPath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", execPath, sizeof(execPath) - 1);
    if (len != -1) {
        execPath[len] = '\0';  // Null-terminate the string
        std::cout << "Executable Path: " << execPath << std::endl;
    } else {
        std::cerr << "Error retrieving executable path" << std::endl;
    }
    return 0;
}