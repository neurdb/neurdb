#ifndef MSG_H
#define MSG_H

#include "postgres.h"
#include "storage/lock.h"
#include "storage/condition_variable.h"

/* ------------------------------------------------------------------------
 * IPC message communication channels inside postgres shared memory.
 * ------------------------------------------------------------------------
 */


#define KV_CHANNEL_BUFSIZE 65536

typedef struct KVChannelShared {
    uint64 head;
    uint64 tail;
    char buffer[KV_CHANNEL_BUFSIZE];
    LWLock lock;
    ConditionVariable cv;
} KVChannelShared;

typedef struct KVChannel {
    char name[NAMEDATALEN];
    bool is_creator;
    KVChannelShared* shared;
} KVChannel;

extern KVChannel* KVChannelInit(const char* name, bool create);
extern void KVChannelDestroy(KVChannel* channel);
extern bool KVChannelPush(KVChannel* channel, const void* data, Size len, bool block);
extern bool KVChannelPop(KVChannel* channel, void* out, Size len, bool block);
extern void PrintChannelContent(KVChannel* channel);

/* ------------------------------------------------------------------------
 * IPC messages.
 * ------------------------------------------------------------------------
 */

 #define MSG_SIZE 1024

/* Operation codes for KV messages */
typedef enum KVOp {
    kv_none = 0,
    kv_open,
    kv_close,
    kv_count,
    kv_put,
    kv_get,
    kv_delete,
    kv_load,
    kv_batch_read,
    kv_cursor_delete,
    kv_start,
    kv_stop
} KVOp;

/* Status codes for KV responses */
typedef enum KVMsgStatus {
    kv_status_none = 0,
    kv_status_ok,
    kv_status_error,
    kv_status_failed
} KVMsgStatus;

/* Forward declare opaque KVChannel */
struct KVChannel;
typedef struct KVChannel KVChannel;

/* KV message header */
typedef struct KVMsgHeader {
    KVOp op;
    Oid relId;
    KVMsgStatus status;
    uint32 respChannel;
    uint64 entitySize;
} KVMsgHeader;

/* Entity read/write callbacks */
typedef void (*EntityWriter)(KVChannel* channel, uint64* offset, void* entity, uint64 size);
typedef void (*EntityReader)(KVChannel* channel, uint64* offset, void* entity, uint64 size);

/* Full KV message structure */
typedef struct KVMsg {
    KVMsgHeader header;
    void* entity;
    EntityReader reader;
    EntityWriter writer;
} KVMsg;

/* Message constructors */
extern KVMsg NewStatusMsg(KVMsgStatus status, uint32 channel_id);
extern KVMsg NewMsg(KVOp op, Oid rel_id);
extern bool KVChannelPushMsg(KVChannel* channel, KVMsg* msg, bool block);
extern bool KVChannelPopMsg(KVChannel* channel, KVMsg* msg, bool block);
extern void PrintKVMsg(const KVMsg* msg);

/* Default entity handlers */
extern void DefaultWriteEntity(KVChannel* channel, uint64* offset, void* entity, uint64 size);
extern void DefaultReadEntity(KVChannel* channel, uint64* offset, void* entity, uint64 size);

#endif /* MSG_H */
