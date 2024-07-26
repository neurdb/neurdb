#include "socketio_nr.h"

#include <stdlib.h>


void nr_socketio_connect_callback(SocketIOClient *client, cJSON *json) {
    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "sid");
    if (cJSON_IsString(data) && (data->valuestring != NULL)) {
        socketio_set_socket_id(client, data->valuestring);
    }
}

void nr_socketio_data_request_callback(SocketIOClient *client, cJSON *json) {
    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
    if (cJSON_IsString(data) && (data->valuestring != NULL)) {

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
