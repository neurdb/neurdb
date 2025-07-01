#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>

#include <rocksdb/c.h>

#define SOCKET_PATH "/tmp/rocksdb.sock"
#define BUFFER_SIZE 1024

int main() {
    int server_fd, client_fd;
    struct sockaddr_un addr;
    char buf[BUFFER_SIZE];

    rocksdb_t* db = NULL;
    rocksdb_options_t* options = NULL;
    char* err = NULL;

    // Setup UNIX socket
    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(server_fd != -1);

    unlink(SOCKET_PATH); 

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    assert(bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) != -1);
    assert(listen(server_fd, 5) != -1);

    printf("RocksDB server listening on %s\n", SOCKET_PATH);

    while (1) {
        client_fd = accept(server_fd, NULL, NULL);
        if (client_fd == -1) continue;

        ssize_t len = read(client_fd, buf, BUFFER_SIZE - 1);
        if (len > 0) {
            buf[len] = '\0';

            if (strncmp(buf, "OPEN ", 5) == 0) {
                if (db) {
                    snprintf(buf, BUFFER_SIZE, "Error AlreadyOpened\n");
                } else {
                    char* path = buf + 5;
                    options = rocksdb_options_create();
                    rocksdb_options_set_create_if_missing(options, 1);
                    db = rocksdb_open(options, path, &err);
                    if (err) {
                        snprintf(buf, BUFFER_SIZE, "Error %s\n", err);
                        free(err); err = NULL;
                    } else {
                        snprintf(buf, BUFFER_SIZE, "OK\n");
                    }
                }
            } else if (strncmp(buf, "CLOSE", 5) == 0) {
                if (db) {
                    rocksdb_close(db);
                    rocksdb_options_destroy(options);
                    db = NULL;
                    options = NULL;
                    snprintf(buf, BUFFER_SIZE, "OK\n");
                } else { 
                    snprintf(buf, BUFFER_SIZE, "Error NotOpened\n");
                }
            } else if (strncmp(buf, "GET ", 4) == 0) {
                if (!db) {
                    snprintf(buf, BUFFER_SIZE, "Error NotOpened\n");
                } else {
                    char* key = buf + 4;
                    size_t val_len;
                    char* val = rocksdb_get(db, rocksdb_readoptions_create(), key, strlen(key), &val_len, &err);
                    if (err || val == NULL) {
                        snprintf(buf, BUFFER_SIZE, "Error NotFound\n");
                        if (err) { free(err); err = NULL; }
                    } else {
                        snprintf(buf, BUFFER_SIZE, "VAL %.*s\n", (int)val_len, val);
                        free(val);
                    }
                }
            } else if (strncmp(buf, "PUT ", 4) == 0) {
                if (!db) {
                    snprintf(buf, BUFFER_SIZE, "Error NotOpened\n");
                } else {
                    char* kv = buf + 4;
                    char* colon = strchr(kv, ':');
                    if (colon) {
                        *colon = '\0';
                        char* key = kv;
                        char* val = colon + 1;
                        rocksdb_put(db, rocksdb_writeoptions_create(), key, strlen(key), val, strlen(val), &err);
                        if (err) {
                            snprintf(buf, BUFFER_SIZE, "Error PutFailed\n");
                            free(err); err = NULL;
                        } else {
                            snprintf(buf, BUFFER_SIZE, "OK\n");
                        }
                    } else {
                        snprintf(buf, BUFFER_SIZE, "Error BadFormat\n");
                    }
                }
            }
        else if (strncmp(buf, "SCAN ", 5) == 0) {
            if (!db) {
                snprintf(buf, BUFFER_SIZE, "Error NotOpened\n");
            } else {
                char* table_id = buf + 5;  // You can skip this if table_id not enforced

                rocksdb_iterator_t* iter = rocksdb_create_iterator(db, rocksdb_readoptions_create());
                rocksdb_iter_seek_to_first(iter);

                // Simple buffer for response (for real systems, use streaming)
                char result[BUFFER_SIZE * 10] = {0};  // Adjust size as needed
                int offset = 0;

                while (rocksdb_iter_valid(iter)) {
                    size_t klen, vlen;
                    const char* key = rocksdb_iter_key(iter, &klen);
                    const char* val = rocksdb_iter_value(iter, &vlen);

                    offset += snprintf(result + offset, sizeof(result) - offset, 
                        "KEY %.*s VAL %.*s\n", (int)klen, key, (int)vlen, val);

                    rocksdb_iter_next(iter);
                }

                snprintf(result + offset, sizeof(result) - offset, "END\n");

                write(client_fd, result, strlen(result));

                rocksdb_iter_destroy(iter);
                continue;
            }

            write(client_fd, buf, strlen(buf));
        } else {
                snprintf(buf, BUFFER_SIZE, "Error UnknownCmd\n");
            }

            write(client_fd, buf, strlen(buf));
        }
        close(client_fd);
    }

    if (db) rocksdb_close(db);
    if (options) rocksdb_options_destroy(options);
    close(server_fd);
    unlink(SOCKET_PATH);

    return 0;
}
