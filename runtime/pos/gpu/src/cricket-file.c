#include "cricket-types.h"
#include "cricket-util.h"
#include "cricket-file.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *dt2str[] = { NULL,        "registers", "pc",   "globals",
                                "stack",     "param",     "heap", "shared",
                                "callstack", NULL };
const char *cricket_file_dt2str(cricketDataType data_type)
{
    if (data_type < CRICKET_DT_LAST)
        return dt2str[(size_t)data_type];
    else
        return NULL;
}

bool cricket_file_read_mem(const char *path, cricketDataType data_type,
                           const char *suffix, void *data, size_t size)
{
    size_t data_size;
    bool ret;

    ret = cricket_file_read_mem_size(path, data_type, suffix, &data, size,
                                     &data_size);

    if (ret == false)
        return ret;

    if (data_size != size) {
        fprintf(stderr, "cricket-file: stored size does not match actual "
                        "size\n");
        return false;
    }
    return ret;
}

bool cricket_file_exists(const char *path, cricketDataType data_type,
                         const char *suffix)
{
    struct stat path_stat = { 0 };
    FILE *fp = NULL;
    char *file_name;
    const char *dt2str = cricket_file_dt2str(data_type);
    if (!path) {
        fprintf(stderr, "cricket-file: path is NULL\n");
        return false;
    }
    if (!dt2str) {
        fprintf(stderr, "cricket-file: datatype does not exist\n");
        return false;
    }
    if (stat(path, &path_stat) != 0) {
        fprintf(stderr, "cricket-file: path \"%s\" does not exist\n", path);
        return false;
    }
    if (!S_ISDIR(path_stat.st_mode)) {
        fprintf(stderr, "cricket-file: file \"%s\" is not a directory\n", path);
        return false;
    }
    if (suffix == NULL) {
        if (asprintf(&file_name, "%s/%s", path, dt2str) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    } else {
        if (asprintf(&file_name, "%s/%s%s", path, dt2str, suffix) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    }
    if (stat(file_name, &path_stat) != 0) {
        free(file_name);
        return false;
    }
    free(file_name);
    return true;
}

bool cricket_file_read_mem_size(const char *path, cricketDataType data_type,
                                const char *suffix, void **data,
                                size_t alloc_size, size_t *size)
{
    struct stat path_stat = { 0 };
    FILE *fp = NULL;
    char *file_name;
    const char *dt2str = cricket_file_dt2str(data_type);
    bool res = false;
    cricketDataType read_dt;
    if (size == NULL) {
        fprintf(stderr, "cricket-file: size is NULL\n");
        return false;
    }
    if (!path) {
        fprintf(stderr, "cricket-file: path is NULL\n");
        return false;
    }
    if (!dt2str) {
        fprintf(stderr, "cricket-file: datatype does not exist\n");
        return false;
    }
    if (stat(path, &path_stat) != 0) {
        fprintf(stderr, "cricket-file: path \"%s\" does not exist\n", path);
        return false;
    }
    if (!S_ISDIR(path_stat.st_mode)) {
        fprintf(stderr, "cricket-file: file \"%s\" is not a directory\n", path);
        return false;
    }
    if (suffix == NULL) {
        if (asprintf(&file_name, "%s/%s", path, dt2str) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    } else {
        if (asprintf(&file_name, "%s/%s%s", path, dt2str, suffix) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    }
    if ((fp = fopen(file_name, "rb")) == NULL) {
        fprintf(stderr, "cricket-file: error while opening file \"%s\"\n",
                file_name);
        free(file_name);
        return false;
    }
    if (fread(&read_dt, sizeof(cricketDataType), 1, fp) != 1) {
        fprintf(stderr, "cricket-file: error reading data-type from \"%s\"\n",
                file_name);
        goto cleanup;
    }
    if (read_dt != data_type) {
        fprintf(stderr, "cricket-file: read dt != given dt. This points to a "
                        "checkpoint inconsistency! (read %u, given %u)\n",
                read_dt, data_type);
        goto cleanup;
    }
    if (fread(size, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "cricket-file: error reading data-size from \"%s\"\n",
                file_name);
        goto cleanup;
    }

    if (*data == NULL) {
        *data = malloc(*size);
    } else if (*size > alloc_size) {
        fprintf(stderr, "cricket-file: allocated size to small. require at "
                        "least %lu bytes.\n",
                *size);
        goto cleanup;
    }

    if (fread(*data, 1, *size, fp) != *size) {
        fprintf(stderr, "cricket-file: error reading data from \"%s\"\n",
                file_name);
        goto cleanup;
    }

    res = true;
cleanup:
    free(file_name);
    fclose(fp);
    return res;
}

bool cricket_file_store_mem(const char *path, cricketDataType data_type,
                            const char *suffix, void *data, size_t size)
{
    struct stat path_stat = { 0 };
    FILE *fp = NULL;
    char *file_name;
    const char *dt2str = cricket_file_dt2str(data_type);
    bool res = false;
    size_t data_size = size;
    cricketDataType write_dt = data_type;
    if (!path) {
        fprintf(stderr, "cricket-file: path is NULL\n");
        return false;
    }
    if (!dt2str) {
        fprintf(stderr, "cricket-file: datatype does not exist\n");
        return false;
    }
    if (stat(path, &path_stat) != 0) {
        printf("cricket-file: directory \"%s\" does not exist. Let's create it.\n", path);
        if (mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            fprintf(stderr, "cricket-file: failed to create directory \"%s\"\n");
            return false;
        }

    }
    if (stat(path, &path_stat) != 0) {
        fprintf(stderr, "cricket-file: path \"%s\" does not exist (but we just created it)\n", path);
        cricket_error_unreachable();
        return false;
    }
    if (!S_ISDIR(path_stat.st_mode)) {
        fprintf(stderr, "cricket-file: file \"%s\" is not a directory\n", path);
        return false;
    }
    if (suffix == NULL) {
        if (asprintf(&file_name, "%s/%s", path, dt2str) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    } else {
        if (asprintf(&file_name, "%s/%s%s", path, dt2str, suffix) < 0) {
            fprintf(stderr, "cricket-file: memory allocation failed\n");
            return false;
        }
    }
    if ((fp = fopen(file_name, "w+b")) == NULL) {
        fprintf(stderr, "cricket-file: error while opening file\n");
        free(file_name);
        return false;
    }
    if (fwrite(&write_dt, sizeof(cricketDataType), 1, fp) != 1) {
        fprintf(stderr, "cricket-file: error writing data-type to \"%s\"\n",
                file_name);
        goto cleanup;
    }
    if (fwrite(&data_size, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "cricket-file: error writing data-size to \"%s\"\n",
                file_name);
        goto cleanup;
    }
    if (fwrite(data, 1, size, fp) != size) {
        fprintf(stderr, "cricket-file: error writing data to \"%s\"\n",
                file_name);
        goto cleanup;
    }

    res = true;
cleanup:
    free(file_name);
    fclose(fp);
    return res;
}
