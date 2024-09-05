#include "socketio.h"

#include <sio_client.h>
#include <cstring>

struct SocketIOClient {
  sio::client client;
  std::map<std::string, MessageCallback> callbacks;
  char *socket_id = nullptr;
  BatchQueue *queue = nullptr;
};

SocketIOClient *socketio_client() {
  // create a new queue
  return new SocketIOClient();
}

void socketio_set_queue(SocketIOClient *client, BatchQueue *queue) {
  client->queue = queue;
}

BatchQueue *socketio_get_queue(const SocketIOClient *client) {
  return client->queue;
}

void socketio_set_socket_id(SocketIOClient *client, const char *socket_id) {
  client->socket_id = new char[strlen(socket_id) + 1];
  strcpy(client->socket_id, socket_id);
}

char *socketio_get_socket_id(const SocketIOClient *client) {
  return client->socket_id;
}

void socketio_connect(SocketIOClient *client, const char *url) {
  client->client.connect(url);
}

void socketio_disconnect(SocketIOClient *client) { client->client.close(); }

void socketio_emit(SocketIOClient *client, const char *event,
                   const char *data) {
  const sio::message::ptr msg = sio::string_message::create(data);
  client->client.socket()->emit(event, msg);
}

void socketio_register_callback(SocketIOClient *client, const char *event,
                                MessageCallback callback) {
  client->callbacks[event] = callback;
  client->client.socket()->on(
      event, [client, event](const sio::event &response) {
        // message returned should be a JSON object
        cJSON *json = cJSON_CreateObject();
        for (const auto &item : response.get_message()->get_map()) {
          cJSON_AddStringToObject(json, item.first.c_str(),
                                  item.second->get_string().c_str());
        }

        auto callback =
            client->callbacks.find(event);  // find the callback function
        if (callback != client->callbacks.end() && callback->second) {
          callback->second(client, json);
        }
        // cleanup
        cJSON_Delete(json);
      });
}
