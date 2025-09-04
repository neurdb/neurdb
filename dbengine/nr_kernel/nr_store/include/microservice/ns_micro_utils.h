#ifndef NS_UTILS_H
#define NS_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define NS_SOCKET_OK 0              /* success */
#define NS_SOCKET_ERR -1            /* generic error */
#define NS_SOCKET_EOF -2            /* peer closed connection */
#define NS_SOCKET_TOOLARGE -3       /* frame exceeds max_len */
#define NS_SOCKET_NOMEM -4          /* memory allocation failed */


/**
* Blocking read of exactly n bytes unless error/EOF.
* Returns NS_OK on success, NS_EOF if peer closed, NS_ERR on error.
* @param fd socket file descriptor
* @param buf buffer to read into
* @param n number of bytes to read
*/
int ns_micro_read_n(int fd, void *buf, size_t n);


/**
* Blocking write of exactly n bytes unless error.
* Returns NS_OK on success, NS_ERR on error.
* @param fd socket file descriptor
* @param buf buffer to write from
* @param n number of bytes to write
*/
int ns_micro_write_n(int fd, const void *buf, size_t n);


/**
* Write a framed message: [uint32 big-endian length][body bytes].
* 'body_len' must fit in 32 bits.
* Returns NS_OK or NS_ERR.
* @param fd socket file descriptor
* @param body message body
* @param body_len length of the message body
*/
int ns_micro_frame_write(int fd, const uint8_t *body, uint32_t body_len);


/**
* Read a framed message. Allocates '*out_body' with malloc; caller must free().
* Rejects frames of size 0 or > max_len.
* Returns NS_OK, NS_EOF, NS_TOOLARGE, NS_NOMEM, or NS_ERR.
* @param fd socket file descriptor
* @param out_body pointer to the buffer that will hold the message body
* @param out_len pointer to the length of the message body
*/
int ns_micro_frame_read(int fd, uint8_t **out_body, uint32_t *out_len, uint32_t max_len);

/**
* Create a TCP listening socket bound to (host, port) with 'backlog'.
* Use host = NULL or "0.0.0.0" to bind all interfaces.
* Returns listening fd on success (>=0), or -1 on error.
*/
int ns_micro_listen_tcp(const char *host, uint16_t port, int backlog);

/**
* Connect to TCP (host, port). Blocking connect.
* Returns connected fd on success (>=0), or -1 on error (check errno).
*/
int ns_micro_connect_tcp(const char *host, uint16_t port);

#ifdef __cplusplus
}
#endif

#endif //NS_UTILS_H
