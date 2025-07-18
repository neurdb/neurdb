

typedef struct Conn {
    int socket_fd;    /* UNIX socket to RocksDB server */
} RocksConn;


#define SOCKET_PATH "/tmp/rocksdb.sock"
