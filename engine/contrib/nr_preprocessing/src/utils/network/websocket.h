#ifndef WEBSOCKET_H
#define WEBSOCKET_H


#include <libwebsockets.h>

#include "../data_structure/queue.h"

/******** Websocket ********/
/**
 * Information of a websocket connection
 */
typedef struct {
    struct lws_context *context;
    struct lws *wsi;
    pthread_t thread;
    int interrupted;
    BatchDataQueue queue;
    char sid[256];
} WebsocketInfo;

/**
 * Initialize a websocket connection
 * @param url The url of the python server
 * @param port The port of the python server
 * @param path The path of the websocket
 * @param queue_max_size The maximum size of the batch data queue,
 * this is the queue between the main thread and the websocket thread
 * @return websocket_info The information of the websocket connection
 */
WebsocketInfo *
init_ws_connection(const char *url, const int port, const char *path, size_t queue_max_size);


/******** Thread ********/
/**
 * Start a websocket thread
 * @param ws_info The information of the websocket connection
 * @return int The status of the thread
 */
int
start_ws_thread(WebsocketInfo *ws_info);

/**
 * Stop a websocket thread
 */
int
stop_ws_thread(WebsocketInfo *ws_info);

/******** Message (main thread and websocket thread) ********/
static const char *ML_STAGE[] = {"train", "evaluate", "test", "inference"};

typedef enum {
    TRAIN = 0,
    EVALUATE = 1,
    TEST = 2,
    INFERENCE = 3
} MLStage;

/**
 * Send a batch of data from the main thread to the websocket thread.
 * The data will be temporarily stored in a queue and will be extracted
 * by the websocket thread.
 */
int
send_batch_data(WebsocketInfo *ws_info, const char *dataset_name, MLStage ml_stage, const char *batch_data);

#endif //WEBSOCKET_H
