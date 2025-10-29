/*-------------------------------------------------------------------------
 *
 * http_client.h
 * 		HTTP client interface for query optimizer extension
 *
 *-------------------------------------------------------------------------
 */

#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

/* Function declarations */
extern char *send_optimization_request(const char *server_url, const char *sql_query, int timeout_ms);
extern void init_http_client(void);
extern void cleanup_http_client(void);

#endif /* HTTP_CLIENT_H */
