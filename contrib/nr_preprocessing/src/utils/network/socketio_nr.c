#include "socketio_nr.h"

#include <c.h>
#include <stdlib.h>
#include <string.h>
#include <utils/elog.h>


void nr_socketio_connect_callback(SocketIOClient *client, cJSON *json) {
    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "sid");
    if (cJSON_IsString(data) && (data->valuestring != NULL)) {
        socketio_set_socket_id(client, data->valuestring);
    }
}

void nr_socketio_request_data_callback(SocketIOClient *client, cJSON *json) {
    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data_type");
    if (cJSON_IsString(data) && (data->valuestring != NULL)) {
        elog(INFO, "Dequeuing data...");
        if (strcmp(data->valuestring, "train") == 0) {
            // send the training data
            if (socketio_get_queue(client, TRAIN) == NULL) {
                elog(ERROR, "Queue is NULL");
            }
            char *data = dequeue(socketio_get_queue(client, TRAIN));
            elog(INFO, "Training data dequeued...");
            socketio_emit(client, "batch_data", data);
            elog(INFO, "Training data sent...");
        } else if (strcmp(data->valuestring, "evaluate") == 0) {
            // send the evaluation data
            char *data = dequeue(socketio_get_queue(client, EVALUATE));
            socketio_emit(client, "batch_data", data);
        } else if (strcmp(data->valuestring, "test") == 0) {
            // send the testing data
            char *data = dequeue(socketio_get_queue(client, TEST));
            socketio_emit(client, "batch_data", data);
        } else if (strcmp(data->valuestring, "inference") == 0) {
            // send the inference data
            char *data = dequeue(socketio_get_queue(client, INFERENCE));
            socketio_emit(client, "batch_data", data);
        }
    }
}

void nr_socketio_emit_db_init(SocketIOClient *client, const char *dataset_name, const int nfeat, const int nfield) {
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "dataset_name", dataset_name);
    cJSON_AddNumberToObject(json, "nfeat", nfeat);
    cJSON_AddNumberToObject(json, "nfield", nfield);
    // send the json data
    char *data = cJSON_PrintUnformatted(json);
    socketio_emit(client, "dataset_init", data);
    // clean up
    cJSON_Delete(json);
    free(data);
}

void nr_socketio_emit_batch_data(SocketIOClient *client, const char *dataset_name, const MLStage data_type,
                                 const char *batch_data) {
    elog(INFO, "Emitting data...");
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "dataset_name", dataset_name);
    cJSON_AddStringToObject(json, "ml_stage", ML_STAGE[data_type]);
    cJSON_AddStringToObject(json, "dataset", batch_data);
    // enqueue the data
    enqueue(socketio_get_queue(client, data_type), cJSON_PrintUnformatted(json));
    elog(INFO, "Data enqueued...");
    // clean up
    cJSON_Delete(json);
}
