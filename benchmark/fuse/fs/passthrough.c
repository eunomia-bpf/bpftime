#define FUSE_USE_VERSION 26

#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>

// Structure to store filesystem state
struct passthrough_state {
    char* rootdir;
    pthread_mutex_t lock;
    int op_counts[20];  // Array to store operation counts
};

// Names of operations for logging
const char* op_names[] = {
    "full_path", "getattr", "readdir", "open", "create", 
    "read", "write", "truncate", "flush", "release", "fsync"
};

// Enum for operation types
enum op_type {
    OP_FULL_PATH = 0,
    OP_GETATTR,
    OP_READDIR,
    OP_OPEN,
    OP_CREATE,
    OP_READ,
    OP_WRITE,
    OP_TRUNCATE,
    OP_FLUSH,
    OP_RELEASE,
    OP_FSYNC
};

// Global variables for signal handling and state
static char* mountpoint = NULL;
static int running = 1;
static struct passthrough_state* global_state = NULL;

// Get filesystem state
static struct passthrough_state* get_state() {
    return global_state ? global_state : (struct passthrough_state*) fuse_get_context()->private_data;
}

// Increment operation count
static void increment_op_count(enum op_type op) {
    struct passthrough_state* state = get_state();
    pthread_mutex_lock(&state->lock);
    state->op_counts[op]++;
    pthread_mutex_unlock(&state->lock);
}

// Construct full path
static void full_path(char fpath[PATH_MAX], const char* path) {
    struct passthrough_state* state = get_state();
    increment_op_count(OP_FULL_PATH);
    
    // Remove leading slash if present
    const char* rel_path = path;
    if (path[0] == '/')
        rel_path = path + 1;
    
    // Construct full path
    strcpy(fpath, state->rootdir);
    if (state->rootdir[strlen(state->rootdir) - 1] != '/')
        strcat(fpath, "/");
    strcat(fpath, rel_path);
}

// Implementation of getattr
static int passthrough_getattr(const char *path, struct stat *stbuf) {
    increment_op_count(OP_GETATTR);
    char fpath[PATH_MAX];
    full_path(fpath, path);
    
    // Block access to specific directory
    if (strncmp(fpath, "/home/yunwei/linux/arch", 23) == 0) {
        return -EACCES;
    }
    
    int res = lstat(fpath, stbuf);
    if (res == -1)
        return -errno;
    
    return 0;
}

// Implementation of readdir
static int passthrough_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                              off_t offset, struct fuse_file_info *fi) {
    increment_op_count(OP_READDIR);
    char fpath[PATH_MAX];
    full_path(fpath, path);
    
    // Block access to specific directory
    if (strncmp(fpath, "/home/yunwei/linux/arch", 23) == 0) {
        return -EACCES;
    }
    
    DIR *dp = opendir(fpath);
    if (dp == NULL)
        return -errno;
    
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino = de->d_ino;
        st.st_mode = de->d_type << 12;
        if (filler(buf, de->d_name, &st, 0))
            break;
    }
    
    closedir(dp);
    return 0;
}

// Implementation of open
static int passthrough_open(const char *path, struct fuse_file_info *fi) {
    increment_op_count(OP_OPEN);
    char fpath[PATH_MAX];
    full_path(fpath, path);
    
    // Block access to specific directory
    if (strncmp(fpath, "/home/yunwei/linux/arch", 23) == 0) {
        return -EACCES;
    }
    
    int fd = open(fpath, fi->flags);
    if (fd == -1)
        return -errno;
    
    fi->fh = fd;
    return 0;
}

// Implementation of create
static int passthrough_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
    increment_op_count(OP_CREATE);
    char fpath[PATH_MAX];
    full_path(fpath, path);
    
    int fd = open(fpath, O_CREAT | O_WRONLY | O_TRUNC, mode);
    if (fd == -1)
        return -errno;
    
    fi->fh = fd;
    return 0;
}

// Implementation of read
static int passthrough_read(const char *path, char *buf, size_t size, off_t offset,
                          struct fuse_file_info *fi) {
    increment_op_count(OP_READ);
    int fd = fi->fh;
    
    int res = pread(fd, buf, size, offset);
    if (res == -1)
        return -errno;
    
    return res;
}

// Implementation of write
static int passthrough_write(const char *path, const char *buf, size_t size,
                           off_t offset, struct fuse_file_info *fi) {
    increment_op_count(OP_WRITE);
    int fd = fi->fh;
    
    int res = pwrite(fd, buf, size, offset);
    if (res == -1)
        return -errno;
    
    return res;
}

// Implementation of truncate
static int passthrough_truncate(const char *path, off_t size) {
    increment_op_count(OP_TRUNCATE);
    char fpath[PATH_MAX];
    full_path(fpath, path);
    
    int res = truncate(fpath, size);
    if (res == -1)
        return -errno;
    
    return 0;
}

// Implementation of flush
static int passthrough_flush(const char *path, struct fuse_file_info *fi) {
    increment_op_count(OP_FLUSH);
    return 0;
}

// Implementation of release
static int passthrough_release(const char *path, struct fuse_file_info *fi) {
    increment_op_count(OP_RELEASE);
    close(fi->fh);
    return 0;
}

// Implementation of fsync
static int passthrough_fsync(const char *path, int isdatasync, struct fuse_file_info *fi) {
    increment_op_count(OP_FSYNC);
    int res;
    
    if (isdatasync)
        res = fdatasync(fi->fh);
    else
        res = fsync(fi->fh);
    
    if (res == -1)
        return -errno;
    
    return 0;
}

// Signal handler for Ctrl+C
static void handle_sigint(int sig) {
    printf("\nReceived SIGINT (Ctrl+C). Unmounting filesystem...\n");
    running = 0;
    
    // Print final operation counts before unmounting
    if (global_state) {
        printf("\nFinal operation counts:\n");
        pthread_mutex_lock(&global_state->lock);
        
        for (int i = 0; i < sizeof(op_names) / sizeof(op_names[0]); i++) {
            if (global_state->op_counts[i] > 0) {
                printf("%s: %d\n", op_names[i], global_state->op_counts[i]);
            }
        }
        
        pthread_mutex_unlock(&global_state->lock);
    }
    
    // Now unmount the filesystem
    if (mountpoint) {
        char cmd[PATH_MAX + 64];
        sprintf(cmd, "fusermount -u %s", mountpoint);
        system(cmd);
        printf("Filesystem unmounted from %s\n", mountpoint);
    }
    
    // We don't exit() here because we want the program to clean up properly
}

// Thread function to print operation counts
void* print_op_counts(void* arg) {
    struct passthrough_state* state = (struct passthrough_state*)arg;
    
    while (running) {
        printf("\nOperation counts:\n");
        pthread_mutex_lock(&state->lock);
        
        for (int i = 0; i < sizeof(op_names) / sizeof(op_names[0]); i++) {
            if (state->op_counts[i] > 0) {
                printf("%s: %d\n", op_names[i], state->op_counts[i]);
            }
        }
        
        pthread_mutex_unlock(&state->lock);
        
        // Sleep for 10 seconds or until interrupted
        int seconds_left = 100;
        while (running && seconds_left > 0) {
            sleep(1);
            seconds_left--;
        }
    }
    
    return NULL;
}

// Initialize filesystem
static void* passthrough_init(struct fuse_conn_info *conn) {
    struct passthrough_state* state = get_state();
    
    // Create thread for printing operation counts
    pthread_t thread;
    pthread_create(&thread, NULL, print_op_counts, state);
    
    return state;
}

// Clean up filesystem
static void passthrough_destroy(void* private_data) {
    struct passthrough_state* state = (struct passthrough_state*)private_data;
    free(state->rootdir);
    pthread_mutex_destroy(&state->lock);
    free(state);
}

// FUSE operations structure
static struct fuse_operations passthrough_oper = {
    .getattr  = passthrough_getattr,
    .readdir  = passthrough_readdir,
    .open     = passthrough_open,
    .create   = passthrough_create,
    .read     = passthrough_read,
    .write    = passthrough_write,
    .truncate = passthrough_truncate,
    .flush    = passthrough_flush,
    .release  = passthrough_release,
    .fsync    = passthrough_fsync,
    .init     = passthrough_init,
    .destroy  = passthrough_destroy,
};

// Helper function to unmount the filesystem if it's already mounted
static void unmount_if_mounted(const char* path) {
    char cmd[PATH_MAX + 64];
    sprintf(cmd, "fusermount -u %s 2>/dev/null", path);
    system(cmd);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s SOURCE_DIR MOUNT_POINT\n\n", argv[0]);
        printf("Example:\n");
        printf("  %s /path/to/source /path/to/mountpoint\n", argv[0]);
        return 1;
    }
    
    char* source_dir = argv[1];
    char* mount_point = argv[2];
    
    // Save mountpoint for signal handler
    mountpoint = strdup(mount_point);
    if (!mountpoint) {
        perror("Failed to allocate memory for mountpoint");
        return 1;
    }
    
    // Unmount first if already mounted
    printf("Ensuring mount point is not in use...\n");
    unmount_if_mounted(mountpoint);
    
    // Set up signal handler for Ctrl+C
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("Failed to set up signal handler");
        free(mountpoint);
        return 1;
    }
    
    // Create and initialize state
    struct passthrough_state* state = malloc(sizeof(struct passthrough_state));
    if (!state) {
        perror("Failed to allocate memory for state");
        free(mountpoint);
        return 1;
    }
    
    // Save state globally for signal handler
    global_state = state;
    
    // Get absolute path of root directory
    state->rootdir = realpath(source_dir, NULL);
    if (state->rootdir == NULL) {
        perror("Failed to resolve root directory path");
        free(state);
        free(mountpoint);
        return 1;
    }
    
    // Initialize mutex and operation counts
    pthread_mutex_init(&state->lock, NULL);
    memset(state->op_counts, 0, sizeof(state->op_counts));
    
    printf("Starting FUSE filesystem...\n");
    printf("Source directory: %s\n", state->rootdir);
    printf("Mount point: %s\n", mountpoint);
    printf("Press Ctrl+C to unmount and exit\n");
    
    // Build FUSE arguments
    char* fuse_argv[6];
    int fuse_argc = 0;
    
    fuse_argv[fuse_argc++] = argv[0];        // Program name
    fuse_argv[fuse_argc++] = "-f";           // Foreground
    fuse_argv[fuse_argc++] = "-o";           // Options
    fuse_argv[fuse_argc++] = "nonempty";     // Allow non-empty mount point
    fuse_argv[fuse_argc++] = mountpoint;     // Mount point must be the last arg
    
    // Start FUSE with the correct argument ordering
    int ret = fuse_main(fuse_argc, fuse_argv, &passthrough_oper, state);
    
    // Clean up mountpoint memory if we get here
    free(mountpoint);
    
    return ret;
} 