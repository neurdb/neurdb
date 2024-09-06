#ifndef SOCKETIO_CALLBACK_H
#define SOCKETIO_CALLBACK_H

#include "socketio.h"

/**
 * Callback function for the "connect" event
 */
void nr_socketio_connect_callback(SocketIOClient *client, cJSON *json);

/**
 * Callback function for the "request_data" event
 */
void nr_socketio_request_data_callback(SocketIOClient *client, cJSON *json);

/**
 * Emit the "dataset_init" event
 */
void nr_socketio_emit_db_init(SocketIOClient *client, const char *dataset_name,
                              int nfeat, int nfield, int nbatch, int cache_num);

/**
 * Emit the "batch_data" event
 */
void nr_socketio_emit_batch_data(SocketIOClient *client,
                                 const char *dataset_name,
                                 const char *batch_data);

/**
 * Emit the "force_disconnect" event
 */
void nr_socketio_emit_force_disconnect(SocketIOClient *client);

#endif  // SOCKETIO_CALLBACK_H
