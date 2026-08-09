#include <cstdint>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <chrono>

// Shared memory structure
struct baseline_shared_data {
    char accept_url_prefix[128];
    uint64_t accepted_count;
    uint64_t rejected_count;
};

// Global variables
static bool stop = false;
static const char* SHM_NAME = "/baseline_nginx_filter_shm";
static baseline_shared_data* shared_data = nullptr;
static int shm_fd = -1;

// Signal handler
static void sig_handler(int sig) {
    std::cout << "Received signal " << sig << ", exiting..." << std::endl;
    stop = true;
}

// Create shared memory or open existing
bool init_shared_memory(const char* prefix) {
    // Try to create a new shared memory segment
    shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory: " << strerror(errno) << std::endl;
        return false;
    }

    // Set the size of the shared memory segment
    if (ftruncate(shm_fd, sizeof(baseline_shared_data)) == -1) {
        std::cerr << "Failed to set shared memory size: " << strerror(errno) << std::endl;
        close(shm_fd);
        shm_unlink(SHM_NAME);
        return false;
    }

    // Map the shared memory segment into our address space
    shared_data = (baseline_shared_data*)mmap(
        NULL, 
        sizeof(baseline_shared_data), 
        PROT_READ | PROT_WRITE, 
        MAP_SHARED, 
        shm_fd, 
        0
    );

    if (shared_data == MAP_FAILED) {
        std::cerr << "Failed to map shared memory: " << strerror(errno) << std::endl;
        close(shm_fd);
        shm_unlink(SHM_NAME);
        return false;
    }

    // Initialize shared data
    memset(shared_data, 0, sizeof(baseline_shared_data));
    if (prefix) {
        strncpy(shared_data->accept_url_prefix, prefix, sizeof(shared_data->accept_url_prefix) - 1);
        shared_data->accept_url_prefix[sizeof(shared_data->accept_url_prefix) - 1] = '\0';
    } else {
        strcpy(shared_data->accept_url_prefix, "/");
    }

    return true;
}

// Clean up shared memory
void cleanup_shared_memory() {
    if (shared_data != nullptr && shared_data != MAP_FAILED) {
        munmap(shared_data, sizeof(baseline_shared_data));
        shared_data = nullptr;
    }
    
    if (shm_fd != -1) {
        close(shm_fd);
        shm_fd = -1;
    }
    
    shm_unlink(SHM_NAME);
}

int main(int argc, const char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [URL prefix to accept]" << std::endl;
        return 1;
    }

    // Set up signal handlers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    std::cout << "Initializing baseline controller with prefix: " << argv[1] << std::endl;

    // Initialize shared memory
    if (!init_shared_memory(argv[1])) {
        std::cerr << "Failed to initialize shared memory" << std::endl;
        return 1;
    }

    std::cout << "Shared memory initialized successfully" << std::endl;
    std::cout << "Accepting URLs with prefix: " << shared_data->accept_url_prefix << std::endl;

    // Main loop - periodically print stats and check for exit
    std::cout << "Controller running. Press Ctrl+C to exit." << std::endl;
    uint64_t prev_accepted = 0;
    uint64_t prev_rejected = 0;

    while (!stop) {
        // Print statistics every second if they've changed
        uint64_t curr_accepted = shared_data->accepted_count;
        uint64_t curr_rejected = shared_data->rejected_count;
        
        if (curr_accepted != prev_accepted || curr_rejected != prev_rejected) {
            std::cout << "Stats: Accepted: " << curr_accepted 
                      << " (+" << (curr_accepted - prev_accepted) << ")"
                      << ", Rejected: " << curr_rejected 
                      << " (+" << (curr_rejected - prev_rejected) << ")"
                      << std::endl;
            
            prev_accepted = curr_accepted;
            prev_rejected = curr_rejected;
        }
        
        // Sleep for a bit
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Clean up
    std::cout << "Final stats: Accepted: " << shared_data->accepted_count 
              << ", Rejected: " << shared_data->rejected_count << std::endl;
    
    cleanup_shared_memory();
    std::cout << "Controller exited cleanly" << std::endl;
    
    return 0;
} 