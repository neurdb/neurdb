#include "nram_storage/rocksdb.h"
#include <sys/socket.h>
#include <sys/un.h>
#include "postgres.h"

#define SCAN_READ_CHUNK 4096
#define BUFFER_SIZE (5 * 1024 * 1024)  // 5MB default maximum buff size.

static char scan_msg_buf[BUFFER_SIZE];

RocksConn* rocksconn_open(void) {
    RocksConn* conn = (RocksConn*) palloc0(sizeof(RocksConn));

    conn->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (conn->socket_fd < 0)
        elog(ERROR, "RocksConn: Failed to create socket");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, SOCKET_PATH);

    if (connect(conn->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0)
        elog(ERROR, "RocksConn: Failed to connect to server");

    return conn;
}

void rocksconn_close(RocksConn* conn) {
    if (conn == NULL)
        return;
    close(conn->socket_fd);
    pfree(conn);
}


void rocksengine_scan(RocksConn* conn) {
    snprintf(scan_msg_buf, sizeof(scan_msg_buf), "SCAN dummy_table");

    write(conn->socket_fd, scan_msg_buf, strlen(scan_msg_buf));

    char buf[SCAN_READ_CHUNK + 1];
    ssize_t n;
    printf("SCAN Results:\n");

    for (;;) {
        n = read(conn->socket_fd, buf, SCAN_READ_CHUNK);
        if (n <= 0)
            elog(ERROR, "RocksEngine: SCAN read failed");

        buf[n] = '\0';
        printf("%s", buf);

        if (strstr(buf, "END\n") != NULL)
            break;
    }
}
