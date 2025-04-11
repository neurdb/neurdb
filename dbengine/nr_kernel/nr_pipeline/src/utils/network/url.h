#ifndef URL_H
#define URL_H

/**
 * @brief make an HTTP URL in `http://host:port/path`.
 * @param host hostname
 * @param port port
 * @param path URL path
 * @return 
 */
char *make_http_url(const char *host, const int port, const char *path);

#endif
