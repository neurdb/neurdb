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

bool KVChannelPush(KVChannel* channel, const void* data, Size len, bool block) {
    uint64 used, space, pos, end, part;
    if (len > KV_CHANNEL_BUFSIZE)
        elog(ERROR, "KVChannel: message too large");

    for (;;) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

        used = (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) % KV_CHANNEL_BUFSIZE;
        space = KV_CHANNEL_BUFSIZE - used - 1;

        if (len <= space) {
            pos = channel->shared->tail;
            end = (pos + len) % KV_CHANNEL_BUFSIZE;

            if (pos + len <= KV_CHANNEL_BUFSIZE) {
                // NRAM_TEST_INFO("case 1");
                memcpy(channel->shared->buffer + pos, data, len);
            } else {
                // NRAM_TEST_INFO("case 2");
                part = KV_CHANNEL_BUFSIZE - pos;
                memcpy(channel->shared->buffer + pos, data, part);
                memcpy(channel->shared->buffer, (char*)data + part, len - part);
            }

            channel->shared->tail = end;
            ConditionVariableSignal(&channel->shared->cv);
            LWLockRelease(&channel->shared->lock);
            return true;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            return false;
        }

        ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
        LWLockRelease(&channel->shared->lock);
    }
}

bool KVChannelPop(KVChannel* channel, void* out, Size len, bool block) {
    uint64 used, pos, end, part;
    for (;;) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

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
            ConditionVariableSignal(&channel->shared->cv);
            LWLockRelease(&channel->shared->lock);
            // NRAM_TEST_INFO("Pop succeed??");
            return true;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            return false;
        }

        ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
        LWLockRelease(&channel->shared->lock);
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

    if (size == 0)
        return;

    if (size <= remaining) {
        memcpy(buf + pos, entity, size);
        *offset = (pos + size) % buf_size;
    } else {
        memcpy(buf + pos, entity, remaining);
        memcpy(buf, (char*)entity + remaining, size - remaining);
        *offset = size - remaining;
    }
}


void DefaultReadEntity(KVChannel* channel, uint64* offset, void* entity, uint64 size) {
    char* buf = channel->shared->buffer;
    uint64 buf_size = KV_CHANNEL_BUFSIZE;
    uint64 pos = *offset;
    uint64 remaining = buf_size - pos;

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
