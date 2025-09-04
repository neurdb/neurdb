#include "microservice/ns_micro_utils.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>


int ns_micro_read_n(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *) buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r == 0) return NS_SOCKET_EOF; /* peer closed */
        if (r < 0) {
            if (errno == EINTR) continue; /* retry */
            return NS_SOCKET_ERR;
        }
        got += (size_t) r;
    }
    return NS_SOCKET_OK;
}

int ns_micro_write_n(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *) buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue; /* retry */
            return NS_SOCKET_ERR;
        }
        sent += (size_t) w;
    }
    return NS_SOCKET_OK;
}

int ns_micro_frame_write(int fd, const uint8_t *body, uint32_t body_len) {
    uint32_t len = htonl(body_len);
    if (ns_micro_write_n(fd, &len, sizeof(len)) != NS_SOCKET_OK) {
        return NS_SOCKET_ERR;
    }
    if (body_len == 0) {
        return NS_SOCKET_OK;
    }
    if (ns_micro_write_n(fd, body, body_len) != NS_SOCKET_OK) {
        return NS_SOCKET_ERR;
    }
    return NS_SOCKET_OK;
}

int ns_micro_frame_read(int fd, uint8_t **out_body, uint32_t *out_len, uint32_t max_len) {
    *out_body = NULL;
    *out_len = 0;

    uint32_t len = 0;
    int return_code = ns_micro_read_n(fd, &len, sizeof(len));
    if (return_code != NS_SOCKET_OK) return return_code;

    uint32_t body_len = ntohl(len);
    if (body_len == 0 || (max_len > 0 && body_len > max_len)) {
        return NS_SOCKET_TOOLARGE;
    }

    uint8_t *buf = (uint8_t *) malloc(body_len);
    if (buf == NULL) {
        return NS_SOCKET_NOMEM;
    }

    int rc = ns_micro_read_n(fd, buf, body_len);
    if (rc != NS_SOCKET_OK) {
        free(buf);
        return rc;
    }
    *out_body = buf;
    *out_len = body_len;
    return NS_SOCKET_OK;
}

static int ns_micro_sockaddr_any(const char *host, uint16_t port, struct sockaddr_in *sockaddr) {
    memset(sockaddr, 0, sizeof(*sockaddr));
    sockaddr->sin_family = AF_INET;
    sockaddr->sin_port = htons(port);
    if (host == NULL || strcmp(host, "0.0.0.0") == 0 || strcmp(host, "*") == 0) {
        sockaddr->sin_addr.s_addr = htonl(INADDR_ANY);
        return 0;
    }
    if (inet_pton(AF_INET, host, &sockaddr->sin_addr) != 1) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

int ns_micro_listen_tcp(const char *host, uint16_t port, int backlog) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int one = 1;
    (void) setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    (void) setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif

    struct sockaddr_in sockaddr;
    if (ns_micro_sockaddr_any(host, port, &sockaddr) != 0) {
        close(fd);
        return -1;
    }

    if (bind(fd, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, backlog > 0 ? backlog : 128) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int ns_micro_connect_tcp(const char *host, uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in sockaddr;
    if (ns_micro_sockaddr_any(host, port, &sockaddr) != 0) {
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0) {
        int e = errno;
        close(fd);
        errno = e;
        return -1;
    }
    return fd;
}
