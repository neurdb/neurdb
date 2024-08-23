#ifndef WEBSOCKET_H
#define WEBSOCKET_H

#include <libwebsockets.h>

#include "../data_structure/queue.h"
#include "task.h"


/**
 * NeurDB websocket connection
 */
typedef struct {
    struct lws_context *context;
    struct lws *instance;
    pthread_t thread;
    int interrupted;
    int connnected;
    int setuped;
    int task_acknowledged;
    int completed;
    BatchQueue queue;
    char sid[256];
} NrWebsocket;


// ****************************** Initialization, Connection, Disconnection ******************************

/**
 * Initialize a websocket instance, it does not connect to the server
 * @param url The url of the python server
 * @param port The port of the python server
 * @param path The path of the websocket
 * @param queue_max_size The maximum size of the batch data queue,
 * this is the queue between the main thread and the websocket thread
 * @return websocket_info The information of the websocket connection
 */
NrWebsocket *
nws_initialize(const char *url, int port, const char *path, size_t queue_max_size);

/**
 * Connect the websocket connection to the server
 * @param ws The websocket instance
 * @return int The status of the websocket thread
 */
int
nws_connect(NrWebsocket *ws);

/**
 * Disconnect the websocket connection
 * @param ws The websocket instance
 * @return int The status of the websocket thread
 */
int
nws_disconnect(NrWebsocket *ws);

// ****************************** Message ******************************

static const char *ML_STAGE[] = {"train", "evaluate", "test", "inference"};

typedef enum {
    S_TRAIN = 0,
    S_EVALUATE = 1,
    S_TEST = 2,
    S_INFERENCE = 3
} MLStage;

/**
 * Send a batch of data from the main thread to the websocket thread.
 * The data will be temporarily stored in a queue and will be extracted
 * by the websocket thread.
 * @param ws The websocket instance
 * @param batch_id The id of the batch
 * @param ml_stage The machine learning stage, i.e., train, evaluate, test, or inference
 * @param batch_data The data of the batch
 */
void
nws_send_batch_data(NrWebsocket *ws, int batch_id, MLStage ml_stage, const char *batch_data);

/**
 * Send a task to the server
 * @param ws The websocket instance
 * @param ml_task The machine learning task
 * @param task_spec The task specification, it can be TrainTaskSpec, InferenceTaskSpec, or FinetuneTaskSpec
 */
void
nws_send_task(NrWebsocket *ws, MLTask ml_task, void *task_spec);

void
nws_wait_completion(NrWebsocket *ws);

#endif //WEBSOCKET_H
