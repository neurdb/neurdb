#include "msg.h"
#include "utils/memutils.h"
#include "utils/wait_event.h"
#include "nram_access/kv.h"

KVChannel* KVChannelInit(const char* name, bool create) {
    bool found;
    KVChannel* chan;
    KVChannelShared* shared = (KVChannelShared*) ShmemInitStruct(name, sizeof(KVChannelShared), &found);

    if (create && !found) {
        memset(shared, 0, sizeof(KVChannelShared));
        LWLockInitialize(&shared->lock, LWLockNewTrancheId());
        ConditionVariableInit(&shared->cv);
    } else if (!found) 
        elog(ERROR, "[NRAM] shared memory segment %s not found", name);

    chan = (KVChannel*) MemoryContextAlloc(TopMemoryContext, sizeof(KVChannel));
    strlcpy(chan->name, name, NAMEDATALEN);
    chan->is_creator = create;
    chan->shared = shared;

    return chan;
}

void KVChannelDestroy(KVChannel* channel) {
    pfree(channel);
}

void PrintChannelContent(KVChannel* channel) {
    char buf[KV_CHANNEL_BUFSIZE + 1];
    uint64 used, pos;

    LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);
    used = (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) % KV_CHANNEL_BUFSIZE;
    pos = channel->shared->head;

    NRAM_TEST_INFO("[Channel Dump] (proc %d) head=%lu tail=%lu used=%lu",
        getpid(),
        channel->shared->head, channel->shared->tail, used);

    for (uint64 i = 0; i < used && i < KV_CHANNEL_BUFSIZE; i++) {
        uint64 real_pos = (pos + i) % KV_CHANNEL_BUFSIZE;
        unsigned char c = channel->shared->buffer[real_pos];

        if (c >= 32 && c <= 126)  // printable
            buf[i] = c;
        else
            buf[i] = '#'; // unprintable to #.
    }

    buf[used] = '\0';
    NRAM_TEST_INFO("[Channel Dump] (proc %d) content=\"%s\"",
        getpid(),
        buf);
    LWLockRelease(&channel->shared->lock);
}


bool KVChannelPush(KVChannel* channel, const void* data, Size len, bool block) {
    uint64 used, space, pos, end, part;
    if (len > KV_CHANNEL_BUFSIZE)
        elog(ERROR, "KVChannel: message too large");

    for (;;) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

        used = (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) % KV_CHANNEL_BUFSIZE;
        space = KV_CHANNEL_BUFSIZE - used - 1;
        

        if (len <= space) {
            // NRAM_TEST_INFO("[Push] head=%lu tail=%lu used=%lu space=%lu len=%lu",
            //             channel->shared->head, channel->shared->tail, used, space, len);
            pos = channel->shared->tail;
            end = (pos + len) % KV_CHANNEL_BUFSIZE;

            if (pos + len <= KV_CHANNEL_BUFSIZE) {
                // NRAM_TEST_INFO("[Push] case 1: pos=%lu end=%lu", pos, end);
                memcpy(channel->shared->buffer + pos, data, len);
            } else {
                part = KV_CHANNEL_BUFSIZE - pos;
                // NRAM_TEST_INFO("[Push] case 2: pos=%lu part=%lu end=%lu", pos, part, end);
                memcpy(channel->shared->buffer + pos, data, part);
                memcpy(channel->shared->buffer, (char*)data + part, len - part);
            }

            channel->shared->tail = end;
            // NRAM_TEST_INFO("[Push] Updated head=%lu", channel->shared->head);
            ConditionVariableBroadcast(&channel->shared->cv);
            LWLockRelease(&channel->shared->lock);
            return true;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            return false;
        }

        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);
        ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
    }
}

static bool UnsafeKVChannelPop(KVChannel* channel, void* out, Size len) {
    uint64 used, pos, end, part;
    used = (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) % KV_CHANNEL_BUFSIZE;
    if (len <= used) {
        pos = channel->shared->head;
        end = (pos + len) % KV_CHANNEL_BUFSIZE;

        if (pos + len <= KV_CHANNEL_BUFSIZE) {
            memcpy(out, channel->shared->buffer + pos, len);
        } else {
            part = KV_CHANNEL_BUFSIZE - pos;
            memcpy(out, channel->shared->buffer + pos, part);
            memcpy((char*)out + part, channel->shared->buffer, len - part);
        }
        channel->shared->head = end;
        return true;
    } else {
        return false;
    }
}


bool KVChannelPop(KVChannel* channel, void* out, Size len, bool block) {
    for (;;) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

        if (UnsafeKVChannelPop(channel, out, len)) {
            ConditionVariableBroadcast(&channel->shared->cv);
            LWLockRelease(&channel->shared->lock);
            return true;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            return false;
        }

        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);
        ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
    }
}


KVMsg NewStatusMsg(KVMsgStatus status, uint32 channel_id) {
    KVMsg msg;
    memset(&msg, 0, sizeof(KVMsg));
    msg.header.status = status;
    msg.header.respChannel = channel_id;
    return msg;
}

KVMsg NewMsg(KVOp op, Oid rel_id) {
    KVMsg msg;
    memset(&msg, 0, sizeof(KVMsg));
    msg.header.op = op;
    msg.header.relId = rel_id;
    return msg;
}

void DefaultWriteEntity(KVChannel* channel, uint64* offset, void* entity, uint64 size) {
    char* buf = channel->shared->buffer;
    uint64 buf_size = KV_CHANNEL_BUFSIZE;
    uint64 pos = *offset;
    uint64 remaining = buf_size - pos;

    Assert(pos >= 0 && pos < buf_size);

    if (size == 0)
        return;

    if (size <= remaining) {
        memcpy(buf + pos, entity, size);
    } else {
        memcpy(buf + pos, entity, remaining);
        memcpy(buf, (char*)entity + remaining, size - remaining);
    }

    *offset = (pos + size) % buf_size;    
}


void DefaultReadEntity(KVChannel* channel, uint64* offset, void* entity, uint64 size) {
    char* buf = channel->shared->buffer;
    uint64 buf_size = KV_CHANNEL_BUFSIZE;
    uint64 pos = *offset;
    uint64 remaining = buf_size - pos;

    Assert(pos >= 0 && pos < buf_size);

    if (size == 0)
        return;

    if (size <= remaining) {
        memcpy(entity, buf + pos, size);
        *offset = (pos + size) % buf_size;
    } else {
        memcpy(entity, buf + pos, remaining);
        memcpy((char*)entity + remaining, buf, size - remaining);
        *offset = size - remaining;
    }
}

void PrintKVMsg(const KVMsg* msg) {
    const char* op_str;
    const char* status_str;

    /* Decode operation */
    switch (msg->header.op) {
        case kv_none: op_str = "NONE"; break;
        case kv_open: op_str = "OPEN"; break;
        case kv_close: op_str = "CLOSE"; break;
        case kv_count: op_str = "COUNT"; break;
        case kv_put: op_str = "PUT"; break;
        case kv_get: op_str = "GET"; break;
        case kv_delete: op_str = "DELETE"; break;
        case kv_load: op_str = "LOAD"; break;
        case kv_batch_read: op_str = "BATCH_READ"; break;
        case kv_cursor_delete: op_str = "CURSOR_DELETE"; break;
        case kv_start: op_str = "START"; break;
        case kv_stop: op_str = "STOP"; break;
        default: op_str = "UNKNOWN_OP"; break;
    }

    /* Decode status */
    switch (msg->header.status) {
        case kv_status_none: status_str = "NONE"; break;
        case kv_status_ok: status_str = "OK"; break;
        case kv_status_error: status_str = "ERROR"; break;
        case kv_status_failed: status_str = "FAILED"; break;
        default: status_str = "UNKNOWN_STATUS"; break;
    }

    /* Print header */
    elog(INFO, "KVMsg: op=%s relId=%u status=%s respChannel=%u entitySize=%lu",
         op_str, msg->header.relId, status_str, msg->header.respChannel, msg->header.entitySize);

    /* Print entity if applicable */
    if (msg->entity != NULL && msg->header.entitySize > 0) {
        char preview[256];
        uint64 max_print = msg->header.entitySize;

        if (max_print >= sizeof(preview))
            max_print = sizeof(preview) - 1;

        memcpy(preview, msg->entity, max_print);
        preview[max_print] = '\0';

        elog(INFO, "KVMsg entity preview: \"%s\" (truncated if larger)", preview);
    } else {
        elog(INFO, "KVMsg entity: NULL");
    }
}

bool KVChannelPushMsg(KVChannel* channel, KVMsg* msg, bool block) {
    uint64 header_size = sizeof(KVMsgHeader);
    uint64 entity_size = msg->header.entitySize;
    uint64 total_size = header_size + entity_size;
    char temp[MSG_SIZE] = {0};
    uint64 offset = 0;

    if (total_size > MSG_SIZE) {
        elog(ERROR, "KVChannelPushMsg: total message too large (%lu bytes)", total_size);
        return false;
    }

    Assert(total_size <= MSG_SIZE);
    memcpy(temp + offset, &msg->header, header_size);
    offset += header_size;
    if (entity_size > 0 && msg->entity != NULL) {
        memcpy(temp + offset, msg->entity, entity_size);
        offset += entity_size;
    }

    return KVChannelPush(channel, temp, total_size, block);
}

bool KVChannelPopMsg(KVChannel* channel, KVMsg* msg, bool block) {
    for (;;) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);
        if (UnsafeKVChannelPop(channel, &msg->header, sizeof(KVMsgHeader))) {
            msg->entity = palloc(msg->header.entitySize);
            if (!UnsafeKVChannelPop(channel, msg->entity, msg->header.entitySize)) {
                elog(ERROR, "Corrupted message: the remaining message is smaller than entitySize %lu", 
                    msg->header.entitySize);
            }
            return true;
        }
        if (!block) {
            LWLockRelease(&channel->shared->lock);
            return false;
        }

        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);
        ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
    }
}
