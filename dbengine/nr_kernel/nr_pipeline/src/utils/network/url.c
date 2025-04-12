#include "utils/network/url.h"

#define MAX_URL_LEN 256

char *make_http_url(const char *host, const int port, const char *path) {
  char *url = (char *)malloc(sizeof(char) * MAX_URL_LEN);
  snprintf(url, MAX_URL_LEN, "http://%s:%d%s", host, port, path);
  return url;
}
