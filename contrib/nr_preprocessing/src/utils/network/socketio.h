/**
 * This component is to provide socket-io communication to the flask server.
 * Wrap the socket.io-client-cpp to provide the socket-io communication in C.
 */
#ifndef SOCKETIO_H
#define SOCKETIO_H

#include <cjson/cJSON.h>

#ifdef __cplusplus
extern "C" {
#endif

static const char *ML_STAGE[] = {"train", "evaluate", "test", "inference"};

typedef enum {
    TRAIN = 0,
    EVALUATE = 1,
    TEST = 2,
    INFERENCE = 3
} MLStage;

/**
 * Wrapper for the socket.io client
 */
typedef struct SocketIOClient SocketIOClient;

/**
 * Event callback function
 */
typedef void (*MessageCallback)(SocketIOClient *client, cJSON *message);

/**
 * Create a SocketIOClient instance
 * @return SocketIOClient* The created SocketIOClient instance
 */
SocketIOClient *socketio_client();

/**
 * Set the socket id for the client
 * @param client SocketIOClient* The client to set the socket id
 * @param socket_id char* The socket id to set
 */
void
socketio_set_socket_id(SocketIOClient *client, char *socket_id);

char *
socketio_get_socket_id(const SocketIOClient *client);

/**
 * Connect to the socket.io server
 * @param client SocketIOClient* The client to connect
 * @param url const char* The url of the server
 */
void
socketio_connect(SocketIOClient *client, const char *url);

void
socketio_disconnect(SocketIOClient *client);

/**
 * Emit an event to the server
 * @param client SocketIOClient* The client to emit the event
 * @param event const char* The event to emit
 * @param data const char* The data to emit
 */
void
socketio_emit(SocketIOClient *client, const char *event, const char *data);

/**
 * Register a callback function for the event
 * @param client SocketIOClient* The client to register the callback
 * @param event const char* The event name
 * @param callback MessageCallback Pointer to the callback function
 */
void
socketio_register_callback(SocketIOClient *client, const char *event, MessageCallback callback);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //SOCKETIO_H
