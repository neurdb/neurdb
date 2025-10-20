#ifndef NS_MICRO_PROTO_H
#define NS_MICRO_PROTO_H

#include <stdint.h>
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    // status
    NS_MICRO_TASK_INVALID = 0,
    NS_MICRO_TASK_SHUTDOWN = 1,
    // tasks
    NS_MICRO_TASK_SAVE_MODEL = 10,
    NS_MICRO_TASK_SAVE_MODELS = 11,
    // NS_MICRO_TASK_SAVE_MODELS_FROM_PATHS = 12,
    NS_MICRO_TASK_LOAD_MODEL = 13,
    NS_MICRO_TASK_INFERENCE = 14,
    NS_MICRO_TASK_CLEAN_CACHE = 15,
    // responses
    NS_MICRO_TASK_OK = 1000,
    NS_MICRO_TASK_ERROR = 1001
} NSMicroTaskType;

typedef struct NSMicroTaskMsgHeader {
    uint32_t type; // NSMicroTaskType
    uint32_t resp_id; // response channle ID
    uint32_t payload_size; // payload size in bytes
} NSMicroTaskMsgHeader;

typedef struct NSMicroTaskMsg {
    NSMicroTaskMsgHeader header;
    uint8_t payload[];
} NSMicroTaskMsg;

NSMicroTaskMsg *NewMicroTask(NSMicroTaskType type, uint32_t payload_size, uint32_t resp_id, const void *payload);

NSMicroTaskMsg *NewMicroOkResponse();

NSMicroTaskMsg *NewMicroErrorResponse(const char *errmsg);

size_t ns_micro_msg_size(uint32_t payload_size); // header + payload

void ns_micro_msg_free(NSMicroTaskMsg *message);

int ns_micro_msg_serialize(const NSMicroTaskMsg *message, uint8_t **out, uint32_t *out_len);

NSMicroTaskMsg *ns_micro_msg_deserialize(const uint8_t *in, uint32_t in_len);

#ifdef __cplusplus
}
#endif

#endif //NS_MICRO_PROTO_H
