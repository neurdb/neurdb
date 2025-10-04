#include "microservice/ns_micro_proto.h"

#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>


size_t ns_micro_msg_size(uint32_t payload_size) {
    return sizeof(NSMicroTaskMsgHeader) + static_cast<size_t>(payload_size);
}

void ns_micro_msg_free(NSMicroTaskMsg *message) {
    if (message) {
        free(message);
    }
}

static NSMicroTaskMsg *ns_micro_msg_alloc(uint32_t type, uint32_t resp_id, uint32_t payload_size) {
    const size_t total_size = ns_micro_msg_size(payload_size);
    auto message = static_cast<NSMicroTaskMsg *>(malloc(total_size));
    if (!message) {
        return nullptr;
    }
    message->header.type = type;
    message->header.resp_id = resp_id;
    message->header.payload_size = payload_size;
    return message;
}

NSMicroTaskMsg *NewMicroTask(NSMicroTaskType type, uint32_t payload_size, uint32_t resp_id, const void *payload) {
    NSMicroTaskMsg *message = ns_micro_msg_alloc(static_cast<uint32_t>(type), resp_id, payload_size);
    if (!message) {
        return nullptr;
    }
    if (payload_size > 0 && payload != nullptr) {
        memcpy(message->payload, payload, payload_size);
    }
    return message;
}

NSMicroTaskMsg *NewMicroOkResponse() {
    return ns_micro_msg_alloc((uint32_t) NS_MICRO_TASK_OK, 0, 0);
}

NSMicroTaskMsg *NewMicroErrorResponse(const char *errmsg) {
    uint32_t len = 0;
    if (errmsg) {
        len = static_cast<uint32_t>(strlen(errmsg) + 1);
    }
    NSMicroTaskMsg *message = ns_micro_msg_alloc((uint32_t) NS_MICRO_TASK_ERROR, 0, len);
    if (!message) {
        return nullptr;
    }
    memcpy(message->payload, errmsg, len);
    return message;
}

int ns_micro_msg_serialize(const NSMicroTaskMsg *message, uint8_t **out, uint32_t *out_len) {
    if (!message || !out || !out_len) return -1;

    constexpr uint32_t header_size = (uint32_t) sizeof(NSMicroTaskMsgHeader);
    const uint32_t total_len = header_size + message->header.payload_size;

    auto buf = static_cast<uint8_t *>(malloc(total_len));

    auto ptr = reinterpret_cast<uint32_t *>(buf);
    ptr[0] = htonl(message->header.type);
    ptr[1] = htonl(message->header.resp_id);
    ptr[2] = htonl(message->header.payload_size);

    if (message->header.payload_size) {
        memcpy(buf + header_size, message->payload, message->header.payload_size);
    }
    *out = buf;
    *out_len = total_len;
    return 0;
}

NSMicroTaskMsg *ns_micro_msg_deserialize(const uint8_t *in, uint32_t in_len) {
    if (!in) {
        return nullptr;
    }

    constexpr uint32_t header_size = (uint32_t) sizeof(NSMicroTaskMsgHeader);
    if (in_len < header_size) {
        return nullptr;
    }

    NSMicroTaskMsgHeader header;
    memcpy(&header, in, header_size);

    uint32_t type = ntohl(header.type);
    uint32_t resp_id = ntohl(header.resp_id);
    uint32_t payload_size = ntohl(header.payload_size);

    if (in_len != header_size + payload_size) {
        return nullptr;
    }

    NSMicroTaskMsg *msg = static_cast<NSMicroTaskMsg *>(malloc(header_size + static_cast<size_t>(payload_size)));

    msg->header.type = type;
    msg->header.resp_id = resp_id;
    msg->header.payload_size = payload_size;
    memcpy(msg->payload, in + header_size, payload_size);
    return msg;
}
