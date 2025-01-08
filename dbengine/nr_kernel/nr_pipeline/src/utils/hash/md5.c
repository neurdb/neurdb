#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>

#include "md5.h"

static int compute_md5(const char *str, unsigned char *md5_hash,
                       unsigned int *md5_length) {
  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  if (mdctx == NULL) {
    fprintf(stderr, "EVP_MD_CTX_new failed\n");
    return 0;
  }
  if (EVP_DigestInit_ex(mdctx, EVP_md5(), NULL) != 1) {
    fprintf(stderr, "EVP_DigestInit_ex failed\n");
    EVP_MD_CTX_free(mdctx);
    return 0;
  }
  if (EVP_DigestUpdate(mdctx, str, strlen(str)) != 1) {
    fprintf(stderr, "EVP_DigestUpdate failed\n");
    EVP_MD_CTX_free(mdctx);
    return 0;
  }
  if (EVP_DigestFinal_ex(mdctx, md5_hash, md5_length) != 1) {
    fprintf(stderr, "EVP_DigestFinal_ex failed\n");
    EVP_MD_CTX_free(mdctx);
    return 0;
  }
  EVP_MD_CTX_free(mdctx);
  return 1;
}

char *nr_md5_str(const char *string) {
  unsigned char md5_hash[EVP_MAX_MD_SIZE];
  unsigned int md5_length;

  if (compute_md5(string, md5_hash, &md5_length)) {
    char *result = (char *)malloc(
        md5_length * 2 + 1);  // each byte is represented by 2 hex characters
    for (unsigned int i = 0; i < md5_length; i++) {
      sprintf(&result[i * 2], "%02x", md5_hash[i]);
    }
    result[md5_length * 2] = '\0';  // Null-terminate the string
    return result;
  }
  return NULL;
}

char *nr_md5_list(char **str_list, int list_size) {
  // get the total length of the concatenated string
  int total_length = 0;
  for (int i = 0; i < list_size; i++) {
    total_length += strlen(str_list[i]);
  }

  char *concat_str = (char *)malloc(total_length + 1);
  concat_str[0] = '\0';  // str is null-terminated
  for (int i = 0; i < list_size; i++) {
    strcat(concat_str, str_list[i]);
  }

  char *md5_result = nr_md5_str(concat_str);
  free(concat_str);
  return md5_result;
}
