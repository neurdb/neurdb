#include "http.h"

#include <c.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>


#define SERVER_URL "http://localhost:8090"

/**
 * Resquest the server to train a model
 * @param libsvm_data char* Training data in libsvm format
 * @param batch_size int Batch size in training
 * @param model_name char* Model name TODO: I think here we shouldn't pass the name, but the architecture of the model
 */
void request_train(const char *libsvm_data, const int batch_size, const char *model_name) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        char url[256];
        snprintf(url, sizeof(url), "%s/train", SERVER_URL);

        curl_mime *form = curl_mime_init(curl);
        // set up fields in the form
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "libsvm_data");
        curl_mime_data(field, libsvm_data, CURL_ZERO_TERMINATED);

        field = curl_mime_addpart(form);
        curl_mime_name(field, "batch_size");
        char batch_size_str[10];
        snprintf(batch_size_str, sizeof(batch_size_str), "%d", batch_size);
        curl_mime_data(field, batch_size_str, CURL_ZERO_TERMINATED);

        field = curl_mime_addpart(form);
        curl_mime_name(field, "model_name");
        curl_mime_data(field, model_name, CURL_ZERO_TERMINATED);

        // set the url and form
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            long reponse_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &reponse_code);

            if (reponse_code != 200) {
                // train failed
                fprintf(stderr, "Response code: %ld, failed to train the model\n", reponse_code);
            } else {
                // train success
                printf("Response from the server");
            }
        }
        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}


/**
 * Resquest the server to make a forward inference with a model
 * @param libsvm_data char* Inference data in libsvm format
 * @param model_name char* Model name
 * @param model_id int Trained model id
 * @param batch_size int Batch size in inference
 */
void request_inference(const char* libsvm_data, const char* model_name, const int model_id, const int batch_size) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        char url[256];
        snprintf(url, sizeof(url), "%s/inference", SERVER_URL);

        curl_mime *form = curl_mime_init(curl);
        // set up fields in the form
        curl_mimepart *field = curl_mime_addpart(form);
        curl_mime_name(field, "libsvm_data");
        curl_mime_data(field, libsvm_data, CURL_ZERO_TERMINATED);

        field = curl_mime_addpart(form);
        curl_mime_name(field, "model_name");
        curl_mime_data(field, model_name, CURL_ZERO_TERMINATED);

        field = curl_mime_addpart(form);
        curl_mime_name(field, "model_id");
        char model_id_str[10];
        snprintf(model_id_str, sizeof(model_id_str), "%d", model_id);
        curl_mime_data(field, model_id_str, CURL_ZERO_TERMINATED);

        field = curl_mime_addpart(form);
        curl_mime_name(field, "batch_size");
        char batch_size_str[10];
        snprintf(batch_size_str, sizeof(batch_size_str), "%d", batch_size);
        curl_mime_data(field, batch_size_str, CURL_ZERO_TERMINATED);

        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            long reponse_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &reponse_code);

            if (reponse_code != 200) {
                // inference failed
                fprintf(stderr, "Response code: %ld, failed to make inference\n", reponse_code);
            } else {
                // inference success
                printf("Response from the server");
            }
        }
        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}
