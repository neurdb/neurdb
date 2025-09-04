#ifndef CHANNEL_H
#define CHANNEL_H

#include <postgres.h>
#include <storage/condition_variable.h>
#include <storage/lwlock.h>
#include <stddef.h>

#include "pgext/global.h"


#define NSTASK_HDRSZ  offsetof(NSTaskMsg, payload)

extern int GetNSChannelTrancheId(void);

extern void NSChannelRegisterTranche(void);

typedef enum NSTaskType {
    NS_TASK_NONE = 0,
    NS_TASK_SAVE_MODEL,
    NS_TASK_SAVE_MODELS,
    NS_TASK_SAVE_MODELS_FROM_PATHS,
    NS_TASK_SAVE_MODEL_DRY_RUN,
    NS_TASK_LOAD_MODEL,
    NS_TASK_INFERNECE,
    NS_TASK_SHUTDOWN,
    NS_TASK_CLEAN_CACHE,
    // response
    NS_TASK_OK,
    NS_TASK_ERROR
} NSTaskType;


typedef struct NSChannel {
    char name[NAMEDATALEN];
    uint64 head;
    uint64 tail;
    pg_atomic_uint32 is_active;
    LWLock lock;
    ConditionVariable cv;
    char buffer[NS_CHANNEL_BUFSIZE];
} NSChannel;

typedef struct NSTaskMsg {
    NSTaskType type;
    uint32 payload_size;
    uint32 resp_channel;
    char payload[FLEXIBLE_ARRAY_MEMBER];
} NSTaskMsg;

extern NSChannel* NSChannelInit(const char* name, bool create);
extern void NSChannelDestroy(NSChannel* channel);
extern bool NSChannelPush(NSChannel* channel, const NSTaskMsg* message);
extern NSTaskMsg* NSChannelPop(NSChannel *channel);
extern NSTaskMsg* NewEmptyTask();
extern NSTaskMsg* NewTask(NSTaskType type, uint32 payload_size, uint32 resp_channel, const void *payload);
extern NSTaskMsg* NewOkResponse();
extern NSTaskMsg* NewErrorResponse(const char *error_msg);

#endif //CHANNEL_H
