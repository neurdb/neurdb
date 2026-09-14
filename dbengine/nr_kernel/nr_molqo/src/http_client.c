/*-------------------------------------------------------------------------
 *
 * http_client.c
 * 		HTTP client implementation for query optimizer extension
 *
 * This module provides HTTP client functionality to communicate with
 * external query optimization server.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "utils/memutils.h"
#include "utils/guc.h"

#include <curl/curl.h>
#include <string.h>
#include <json-c/json.h>

/* Response data structure for libcurl */
struct http_response {
    char *data;
    size_t size;
};

/* Callback function for libcurl to write response data */
static size_t
write_callback(void *contents, size_t size, size_t nmemb, struct http_response *response)
{
    size_t realsize = size * nmemb;
    char *ptr = repalloc(response->data, response->size + realsize + 1);

    if (!ptr)
        return 0; /* out of memory */

    response->data = ptr;
    memcpy(&(response->data[response->size]), contents, realsize);
    response->size += realsize;
    response->data[response->size] = 0; /* null terminate */

    return realsize;
}

/*
 * Send HTTP POST request with SQL query to optimization server
 */
char *
send_optimization_request(const char *server_url, const char *sql_query, int timeout_ms)
{
    CURL *curl;
    CURLcode res;
    struct http_response response = {0};
    struct curl_slist *headers = NULL;
    json_object *json_request = NULL;
    json_object *json_sql = NULL;
    json_object *json_response = NULL;
    json_object *json_optimized_sql = NULL;
    const char *json_string;
    char *result = NULL;

    /* Initialize response buffer */
    response.data = palloc(1);
    response.size = 0;

    /* Initialize libcurl */
    curl = curl_easy_init();
    if (!curl)
    {
        ereport(WARNING,
                (errmsg("Failed to initialize HTTP client")));
        pfree(response.data);
        return NULL;
    }

    PG_TRY();
    {
        /* Create JSON request payload */
        json_request = json_object_new_object();
        json_sql = json_object_new_string(sql_query);
        json_object_object_add(json_request, "sql", json_sql);
        json_string = json_object_to_json_string(json_request);

        /* Set HTTP headers */
        headers = curl_slist_append(headers, "Content-Type: application/json");

        /* Configure curl options */
        curl_easy_setopt(curl, CURLOPT_URL, server_url);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "PostgreSQL-QueryOptimizer/1.0");

        /* Perform the request */
        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
        {
            ereport(WARNING,
                    (errmsg("HTTP request failed: %s", curl_easy_strerror(res))));
        }
        else
        {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

            if (response_code == 200 && response.data && response.size > 0)
            {
                /* Parse JSON response */
                json_response = json_tokener_parse(response.data);
                if (json_response)
                {
                    if (json_object_object_get_ex(json_response, "optimized_sql", &json_optimized_sql))
                    {
                        const char *optimized_sql = json_object_get_string(json_optimized_sql);
                        if (optimized_sql && strlen(optimized_sql) > 0)
                        {
                            result = pstrdup(optimized_sql);
                        }
                    }
                }
                else
                {
                    ereport(WARNING,
                            (errmsg("Invalid JSON response from optimization server")));
                }
            }
            else
            {
                ereport(WARNING,
                        (errmsg("Optimization server returned HTTP %ld", response_code)));
            }
        }
    }
    PG_CATCH();
    {
        /* Clean up on error */
        if (json_response)
            json_object_put(json_response);
        if (json_request)
            json_object_put(json_request);
        if (headers)
            curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        if (response.data)
            pfree(response.data);
        PG_RE_THROW();
    }
    PG_END_TRY();

    /* Clean up */
    if (json_response)
        json_object_put(json_response);
    if (json_request)
        json_object_put(json_request);
    if (headers)
        curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (response.data)
        pfree(response.data);

    return result;
}

/*
 * Initialize HTTP client library
 */
void
init_http_client(void)
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

/*
 * Cleanup HTTP client library
 */
void
cleanup_http_client(void)
{
    curl_global_cleanup();
}
