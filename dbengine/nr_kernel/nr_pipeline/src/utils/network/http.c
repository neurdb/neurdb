#include "http.h"

#include <c.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <utils/elog.h>

#include "interface.h"
#include "utils/network/url.h"

void *send_train_task(TrainingInfo *info) {
  const char *model_name = info->model_name;
  const char *table_name = info->table_name;
  const char *client_socket_id = info->client_socket_id;
  const int batch_size = info->batch_size;
  const int epoch = info->epoch;
  const int train_batch_num = info->train_batch_num;
  const int eva_batch_num = info->eva_batch_num;
  const int test_batch_num = info->test_batch_num;
  const char *features = info->features;
  const char *target = info->target;

  curl_global_init(CURL_GLOBAL_ALL);
  CURL *curl = curl_easy_init();

  if (curl) {
    char *url = make_http_url(NrAIEngineHost, NrAIEnginePort, "/train");

    curl_mime *form = curl_mime_init(curl);

    // Add model_name field
    curl_mimepart *field = curl_mime_addpart(form);
    curl_mime_name(field, "model_name");
    curl_mime_data(field, model_name, CURL_ZERO_TERMINATED);

    // Add table_name field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "table_name");
    curl_mime_data(field, table_name, CURL_ZERO_TERMINATED);

    // Add client_socket_id field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "client_socket_id");
    curl_mime_data(field, client_socket_id, CURL_ZERO_TERMINATED);

    // Add batch_size field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "batch_size");
    char batch_size_str[10];
    snprintf(batch_size_str, sizeof(batch_size_str), "%d", batch_size);
    curl_mime_data(field, batch_size_str, CURL_ZERO_TERMINATED);

    // Add epoch field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "epoch");
    char epoch_str[10];
    snprintf(epoch_str, sizeof(epoch_str), "%d", epoch);
    curl_mime_data(field, epoch_str, CURL_ZERO_TERMINATED);

    // Add train_batch_num field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "train_batch_num");
    char train_batch_num_str[10];
    snprintf(train_batch_num_str, sizeof(train_batch_num_str), "%d",
             train_batch_num);
    curl_mime_data(field, train_batch_num_str, CURL_ZERO_TERMINATED);

    // Add eva_batch_num field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "eva_batch_num");
    char eva_batch_num_str[10];
    snprintf(eva_batch_num_str, sizeof(eva_batch_num_str), "%d", eva_batch_num);
    curl_mime_data(field, eva_batch_num_str, CURL_ZERO_TERMINATED);

    // Add test_batch_num field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "test_batch_num");
    char test_batch_num_str[10];
    snprintf(test_batch_num_str, sizeof(test_batch_num_str), "%d",
             test_batch_num);
    curl_mime_data(field, test_batch_num_str, CURL_ZERO_TERMINATED);

    // Add features field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "features");
    curl_mime_data(field, features, CURL_ZERO_TERMINATED);

    // Add target field
    field = curl_mime_addpart(form);
    curl_mime_name(field, "target");
    curl_mime_data(field, target, CURL_ZERO_TERMINATED);

    // Set the URL and form
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    elog(NOTICE, "Train task sent to server: %s", url);
    const CURLcode res = curl_easy_perform(curl);
    elog(NOTICE, "Train task completed");
    // TODO: here we skip the response code check, need to add them later
    // if (res != CURLE_OK) {
    //     fprintf(stderr, "curl_easy_perform() failed: %s\n",
    //     curl_easy_strerror(res));
    // } else {
    //     long reponse_code;
    //     curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &reponse_code);
    //
    //     if (reponse_code != 200) {
    //         // train failed
    //         fprintf(stderr, "Response code: %ld, failed to send the train
    //         task\n", reponse_code);
    //     } else {
    //         // train success
    //         printf("Response from the server: training task received\n");
    //     }
    // }
    curl_mime_free(form);
    curl_easy_cleanup(curl);

    free(url);
  }
  curl_global_cleanup();
  return NULL;
}

void *send_inference_task(InferenceInfo *info) {
  const char *model_name = info->model_name;
  const int model_id = info->model_id;
  const char *table_name = info->table_name;
  const char *client_socket_id = info->client_socket_id;
  const int batch_size = info->batch_size;
  const int batch_num = info->batch_num;
  curl_global_init(CURL_GLOBAL_ALL);
  CURL *curl = curl_easy_init();

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();

  if (curl) {
    char *url = make_http_url(NrAIEngineHost, NrAIEnginePort, "/inference");

    curl_mime *form = curl_mime_init(curl);
    // set up fields in the form
    curl_mimepart *field = curl_mime_addpart(form);
    curl_mime_name(field, "model_name");
    curl_mime_data(field, model_name, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "model_id");
    char model_id_str[10];
    snprintf(model_id_str, sizeof(model_id_str), "%d", model_id);
    curl_mime_data(field, model_id_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "table_name");
    curl_mime_data(field, table_name, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "client_socket_id");
    curl_mime_data(field, client_socket_id, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "batch_size");
    char batch_size_str[10];
    snprintf(batch_size_str, sizeof(batch_size_str), "%d", batch_size);
    curl_mime_data(field, batch_size_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "batch_num");
    char batch_num_str[10];
    snprintf(batch_num_str, sizeof(batch_num_str), "%d", batch_num);
    curl_mime_data(field, batch_num_str, CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    CURLcode res = curl_easy_perform(curl);
    elog(NOTICE, "Inference task sent to server: %s", url);

    // if (res != CURLE_OK) {
    //     fprintf(stderr, "curl_easy_perform() failed: %s\n",
    //     curl_easy_strerror(res));
    // } else {
    //     long reponse_code;
    //     curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &reponse_code);
    //
    //     if (reponse_code != 200) {
    //         // inference failed
    //         fprintf(stderr, "Response code: %ld, failed to make inference\n",
    //         reponse_code);
    //     } else {
    //         // inference success
    //         printf("Response from the server");
    //     }
    // }
    curl_mime_free(form);
    curl_easy_cleanup(curl);

    free(url);
  }
  curl_global_cleanup();
  return NULL;
}

void *send_finetune_task(FinetuneInfo *info) {
  const char *model_name = info->model_name;
  const int model_id = info->model_id;
  const char *table_name = info->table_name;
  const char *client_socket_id = info->client_socket_id;
  const int batch_size = info->batch_size;
  const int epoch = info->epoch;
  const int train_batch_num = info->train_batch_num;
  const int eva_batch_num = info->eva_batch_num;
  const int test_batch_num = info->test_batch_num;

  curl_global_init(CURL_GLOBAL_ALL);
  CURL *curl = curl_easy_init();

  if (curl) {
    char *url = make_http_url(NrAIEngineHost, NrAIEnginePort, "/finetune");

    curl_mime *form = curl_mime_init(curl);
    // set up fields in the form
    curl_mimepart *field = curl_mime_addpart(form);
    curl_mime_name(field, "model_name");
    curl_mime_data(field, model_name, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "model_id");
    char model_id_str[10];
    snprintf(model_id_str, sizeof(model_id_str), "%d", model_id);
    curl_mime_data(field, model_id_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "table_name");
    curl_mime_data(field, table_name, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "client_socket_id");
    curl_mime_data(field, client_socket_id, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "batch_size");
    char batch_size_str[10];
    snprintf(batch_size_str, sizeof(batch_size_str), "%d", batch_size);
    curl_mime_data(field, batch_size_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "epoch");
    char epoch_str[10];
    snprintf(epoch_str, sizeof(epoch_str), "%d", epoch);
    curl_mime_data(field, epoch_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "train_batch_num");
    char train_batch_num_str[10];
    snprintf(train_batch_num_str, sizeof(train_batch_num_str), "%d",
             train_batch_num);
    curl_mime_data(field, train_batch_num_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "eva_batch_num");
    char eva_batch_num_str[10];
    snprintf(eva_batch_num_str, sizeof(eva_batch_num_str), "%d", eva_batch_num);
    curl_mime_data(field, eva_batch_num_str, CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "test_batch_num");
    char test_batch_num_str[10];
    snprintf(test_batch_num_str, sizeof(test_batch_num_str), "%d",
             test_batch_num);
    curl_mime_data(field, test_batch_num_str, CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    CURLcode res = curl_easy_perform(curl);
    elog(NOTICE, "Finetune task sent to server: %s", url);

    // if (res != CURLE_OK) {
    //     fprintf(stderr, "curl_easy_perform() failed: %s\n",
    //     curl_easy_strerror(res));
    // } else {
    //     long reponse_code;
    //     curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &reponse_code);
    //
    //     if (reponse_code != 200) {
    //         // finetune failed
    //         fprintf(stderr, "Response code: %ld, failed to finetune the
    //         model\n", reponse_code);
    //     } else {
    //         // finetune success
    //         printf("Response from the server");
    //     }
    // }
    curl_mime_free(form);
    curl_easy_cleanup(curl);

    free(url);
  }
  curl_global_cleanup();
  return NULL;
}
