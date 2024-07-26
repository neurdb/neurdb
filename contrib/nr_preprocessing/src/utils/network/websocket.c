#include "websocket.h"
#include "../cjson/cJSON.h"


// ****************************** Queue ******************************
static void init_batch_data_queue(BatchDataQueue *queue, size_t max_size) {
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;
    queue->max_size = max_size;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->consume, NULL);
    pthread_cond_init(&queue->produce, NULL);
}

static void destroy_batch_data_queue(BatchDataQueue *queue) {
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->consume);
    pthread_cond_destroy(&queue->produce);
}

static void enqueue(BatchDataQueue *queue, const char *batch_data) {
    BatchDataNode *node = (BatchDataNode *) malloc(sizeof(BatchDataNode));
    node->batched_data = strdup(batch_data);
    node->next = NULL;

    pthread_mutex_lock(&queue->mutex); // lock the mutex
    while (queue->size >= queue->max_size) {
        // wait if the queue is full
        pthread_cond_wait(&queue->produce, &queue->mutex);
    }
    // LOCKED
    if (queue->tail) {
        queue->tail->next = node;
    } else {
        queue->head = node;
    }
    queue->tail = node;
    queue->size++;
    pthread_cond_signal(&queue->consume); // signal the consumer
    pthread_mutex_unlock(&queue->mutex); // unlock the mutex
}

static char *dequeue(BatchDataQueue *queue) {
    pthread_mutex_lock(&queue->mutex); // lock the mutex
    while (queue->head == NULL) {
        // wait if the queue is empty
        pthread_cond_wait(&queue->consume, &queue->mutex);
    }
    // LOCKED
    BatchDataNode *node = queue->head;
    queue->head = node->next;
    if (queue->head == NULL) {
        queue->tail = NULL;
    }
    queue->size--;
    pthread_cond_signal(&queue->produce); // signal the producer
    pthread_mutex_unlock(&queue->mutex); // unlock the mutex
    // extract the data
    char *data = node->batched_data;
    free(node);
    return data;
}

// ****************************** Websocket ******************************
static int callback_ws(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {
    WebsocketInfo *ws_info = (WebsocketInfo *) lws_wsi_user(wsi);
    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            // connection established
            lwsl_user("Client has connected to the server\n");
            break;
        case LWS_CALLBACK_CLIENT_RECEIVE:
            // received data from the server
            cJSON *json = cJSON_Parse((char *) in);
            if (json == NULL) {
                lwsl_err("Failed to parse the received data\n");
                break;
            }
            // check if the JSON object has the key "data"
            cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
            if (cJSON_IsString(data) && (data->valuestring != NULL)) {
                // store sid
                strncpy(ws_info->sid, data->valuestring, sizeof(ws_info->sid));
                ws_info->sid[sizeof(ws_info->sid) - 1] = '\0';
            }
            // check if the JSON object has the key "key"
            cJSON *key = cJSON_GetObjectItemCaseSensitive(json, "key");
            if (cJSON_IsString(key) && (key->valuestring != NULL)) {
                // TODO: Finish this
            }
            break;
        case LWS_CALLBACK_CLIENT_WRITEABLE:
            // send data to the server
            lwsl_user("Client is ready to send data to the server\n");
            char *batch_data = dequeue(&ws_info->queue);
            if (batch_data) {
                lws_write(wsi, (unsigned char *) batch_data, strlen(batch_data), LWS_WRITE_TEXT);
                free(batch_data);
            }
            if (ws_info->queue.size > 0) {
                lws_callback_on_writable(wsi); // this is done to keep the connection writable
            }
            break;
        case LWS_CALLBACK_CLOSED:
            // connection closed
            lwsl_user("Client has disconnected from the server\n");
            ws_info->interrupted = 1;
            break;
        default:
            break;
    }
    return 0;
}

// define the protocol in the websocket
static const struct lws_protocols protocols[] = {
    {
        "nr-protocol",
        callback_ws,
        0,
        0,
    },
    {NULL, NULL, 0, 0}
};

WebsocketInfo *init_ws_connection(const char *url, const int port, const char *path, const size_t queue_max_size) {
    WebsocketInfo *ws_info = (WebsocketInfo *) malloc(sizeof(WebsocketInfo));
    memset(ws_info, 0, sizeof(WebsocketInfo));

    init_batch_data_queue(&ws_info->queue, queue_max_size);

    struct lws_context_creation_info info = {0};
    info.port = CONTEXT_PORT_NO_LISTEN; // client side, no need to listen
    info.protocols = protocols;
    info.options = LWS_SERVER_OPTION_VALIDATE_UTF8;

    ws_info->context = lws_create_context(&info);
    if (ws_info->context == NULL) {
        lwsl_err("Failed to create websocket context\n");
        lws_context_destroy(ws_info->context);
        free(ws_info);
        return NULL;
    }

    struct lws_client_connect_info connect_info = {0};
    connect_info.context = ws_info->context;
    connect_info.address = url;
    connect_info.port = port;
    connect_info.path = path;
    connect_info.host = lws_canonical_hostname(ws_info->context);
    connect_info.origin = lws_canonical_hostname(ws_info->context); // localhost connection
    connect_info.protocol = protocols[0].name;
    connect_info.pwsi = &ws_info->wsi;
    connect_info.userdata = ws_info;

    ws_info->wsi = lws_client_connect_via_info(&connect_info);
    if (ws_info->wsi == NULL) {
        lwsl_err("Failed to create websocket connection\n");
        lws_context_destroy(ws_info->context);
        free(ws_info);
        return NULL;
    }
    return ws_info;
}


// ****************************** Thread ******************************
static void *websocket_threading(void *arg) {
    WebsocketInfo *ws_info = (WebsocketInfo *) arg;
    while (!ws_info->interrupted) {
        lws_service(ws_info->context, 1000); // serving
    }
    return NULL;
}

int start_ws_thread(WebsocketInfo *ws_info) {
    if (pthread_create(&ws_info->thread, NULL, (void *(*)(void *)) websocket_threading, ws_info) != 0) {
        lwsl_err("Failed to create websocket thread\n");
        lws_context_destroy(ws_info->context);
        free(ws_info);
        return -1;
    }
    return 0;
}

int stop_ws_thread(WebsocketInfo *ws_info) {
    ws_info->interrupted = 1;
    pthread_join(ws_info->thread, NULL);
    lws_context_destroy(ws_info->context);
    destroy_batch_data_queue(&ws_info->queue);
    free(ws_info);
    return 0;
}

// ****************************** Message ******************************
int send_batch_data(WebsocketInfo *ws_info, const char *dataset_name, const MLStage ml_stage, const char *batch_data) {
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "dataset_name", dataset_name);
    cJSON_AddStringToObject(json, "ml_stage", ML_STAGE[ml_stage]);
    cJSON_AddStringToObject(json, "data", batch_data);
    char *data = cJSON_PrintUnformatted(json);
    enqueue(&ws_info->queue, data); // enqueue the data
    // clean up
    cJSON_Delete(json);
    free(data);
    lws_callback_on_writable(ws_info->wsi);
    return 0;
}
