#include "msg.h"
#include "utils/memutils.h"
#include "utils/wait_event.h"
#include "nram_access/kv.h"
#include "portability/instr_time.h"
#include "storage/pmsignal.h"

KVChannel* KVChannelInit(const char* name, bool create) {
    bool found;
    KVChannel* chan;
    KVChannelShared* shared = (KVChannelShared*)ShmemInitStruct(
        name, sizeof(KVChannelShared), &found);
    NRAM_TEST_INFO("Initializing channel %s from proc %d", name, MyProcPid);

    if (create) {
        memset(shared, 0, sizeof(KVChannelShared));
        LWLockInitialize(&shared->lock, LWLockNewTrancheId());
        ConditionVariableInit(&shared->cv);
        pg_atomic_init_u32(&shared->is_running, 1);
    } else if (!found) {
        elog(ERROR, "[NRAM] shared memory segment %s not found", name);
    }

    chan = (KVChannel*)MemoryContextAlloc(TopMemoryContext, sizeof(KVChannel));
    strlcpy(chan->name, name, NAMEDATALEN);
    chan->is_creator = create;
    chan->shared = shared;

    return chan;
}

void KVChannelDestroy(KVChannel* chan) {
    if (chan == NULL) return;
    TerminateChannel(chan);
    pfree(chan);  // Free local memory
}

void PrintChannelContent(KVChannel* channel) {
    char buf[KV_CHANNEL_BUFSIZE + 1];
    uint64 used, pos;

    LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);
    used =
        (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) %
        KV_CHANNEL_BUFSIZE;
    pos = channel->shared->head;

    elog(INFO, "[Channel Dump %s] (proc %d) status=%u head=%lu tail=%lu used=%lu",
                   channel->name, MyProcPid, pg_atomic_read_u32(&channel->shared->is_running), channel->shared->head, channel->shared->tail,
                   used);

    for (uint64 i = 0; i < used && i < KV_CHANNEL_BUFSIZE; i++) {
        uint64 real_pos = (pos + i) % KV_CHANNEL_BUFSIZE;
        unsigned char c = channel->shared->buffer[real_pos];

        if (c >= 32 && c <= 126)  // printable
            buf[i] = c;
        else
            buf[i] = '#';  // unprintable to #.
    }

    buf[used] = '\0';
    elog(INFO, "[Channel Dump %s] (proc %d) content=\"%s\"", channel->name, MyProcPid, buf);
    LWLockRelease(&channel->shared->lock);
}

bool KVChannelPush(KVChannel* channel, const void* data, Size len, long timeout_ms) {
    uint64 used, space, pos, end, part;
    instr_time start_time, cur_time;
    long cur_timeout = timeout_ms;
    // NRAM_INFO();
    bool block = timeout_ms != 0;

    if (len > KV_CHANNEL_BUFSIZE) elog(ERROR, "KVChannel: message too large");

    if (block && timeout_ms > 0) INSTR_TIME_SET_CURRENT(start_time);

    while (pg_atomic_read_u32(&channel->shared->is_running)) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

        used = (channel->shared->tail + KV_CHANNEL_BUFSIZE -
                channel->shared->head) %
               KV_CHANNEL_BUFSIZE;
        space = KV_CHANNEL_BUFSIZE - used - 1;

        if (len <= space) {
            NRAM_TEST_INFO(
                "[Push %s] head=%lu tail=%lu used=%lu space=%lu len=%lu",
                channel->name, channel->shared->head, channel->shared->tail, used, space, len);

            pos = channel->shared->tail;
            end = (pos + len) % KV_CHANNEL_BUFSIZE;

            if (pos + len <= KV_CHANNEL_BUFSIZE) {
                NRAM_TEST_INFO("[Push] case 1: pos=%lu end=%lu", pos, end);
                memcpy(channel->shared->buffer + pos, data, len);
            } else {
                part = KV_CHANNEL_BUFSIZE - pos;
                NRAM_TEST_INFO("[Push] case 2: pos=%lu part=%lu end=%lu", pos,
                               part, end);
                memcpy(channel->shared->buffer + pos, data, part);
                memcpy(channel->shared->buffer, (char*)data + part, len - part);
            }

            channel->shared->tail = end;
            NRAM_TEST_INFO(
                "[Push %s] afterwise head=%lu tail=%lu used=%lu space=%lu len=%lu",
                channel->name, channel->shared->head, channel->shared->tail, used, space, len);

            ConditionVariableBroadcast(&channel->shared->cv);
            LWLockRelease(&channel->shared->lock);
            return true;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            elog(INFO, "KVChannelPush non-blocking and full, drop message");
            return false;
        }

        if (timeout_ms > 0) {
            INSTR_TIME_SET_CURRENT(cur_time);
            INSTR_TIME_SUBTRACT(cur_time, start_time);
            cur_timeout = timeout_ms - (long)INSTR_TIME_GET_MILLISEC(cur_time);

            if (cur_timeout <= 0) {
                LWLockRelease(&channel->shared->lock);
                elog(ERROR, "KVChannelPush encounters timeout");
                return false;
            }
        } else
            cur_timeout = -1;  // Infinite block allowed

        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);

        if (!PostmasterIsAlive())
            proc_exit(0);

        if (cur_timeout > 0)
            ConditionVariableTimedSleep(&channel->shared->cv, cur_timeout, WAIT_EVENT_KV_CHANNEL);
        else
            ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
        ConditionVariableCancelSleep();
    }

    elog(INFO, "KVChannelPush channel not running, drop message");
    return false;
}

static bool UnsafeKVChannelPop(KVChannel* channel, void* out, Size len) {
    uint64 used, pos, end, part;
    used =
        (channel->shared->tail + KV_CHANNEL_BUFSIZE - channel->shared->head) %
        KV_CHANNEL_BUFSIZE;
    if (len <= used) {
        NRAM_TEST_INFO(
            "[Pop %s] head=%lu tail=%lu used=%lu space=%lu len=%lu",
            channel->name, channel->shared->head, channel->shared->tail, used, KV_CHANNEL_BUFSIZE-used, len);
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
        used -= len;
        NRAM_TEST_INFO(
            "[Pop %s] afterwise: head=%lu tail=%lu used=%lu space=%lu len=%lu",
            channel->name, channel->shared->head, channel->shared->tail, used, KV_CHANNEL_BUFSIZE-used, len);
        return true;
    } else {
        return false;
    }
}

bool KVChannelPop(KVChannel* channel, void* out, Size len, long timeout_ms) {
    instr_time start_time, cur_time;
    long cur_timeout = timeout_ms;
    bool block = timeout_ms != 0;
    // NRAM_INFO();

    if (len > KV_CHANNEL_BUFSIZE) elog(ERROR, "KVChannel: pop too much");

    if (block && timeout_ms > 0) INSTR_TIME_SET_CURRENT(start_time);

    while (pg_atomic_read_u32(&channel->shared->is_running)) {
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

        /* Timeout management */
        if (timeout_ms > 0) {
            INSTR_TIME_SET_CURRENT(cur_time);
            INSTR_TIME_SUBTRACT(cur_time, start_time);
            cur_timeout = timeout_ms - (long)INSTR_TIME_GET_MILLISEC(cur_time);

            if (cur_timeout <= 0) {
                LWLockRelease(&channel->shared->lock);
                elog(ERROR, "KVChannelPop encounters timeout");
                return false;
            }
        } else
            cur_timeout = -1;  // No timeout, block indefinitely

        /* Sleep with precise remaining time */
        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);

        if (!PostmasterIsAlive())
            proc_exit(0);

        if (cur_timeout > 0)
            ConditionVariableTimedSleep(&channel->shared->cv, cur_timeout, WAIT_EVENT_KV_CHANNEL);
        else
            ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
        ConditionVariableCancelSleep();
    }

    return false;
}

KVMsg* NewMsg(KVOp op, Oid rel_id, KVMsgStatus status, uint32 channel_id) {
    KVMsg* msg = palloc0(sizeof(KVMsg));
    msg->header.op = op;
    msg->header.relId = rel_id;
    msg->header.status = status;
    msg->header.respChannel = channel_id;

    msg->entity = NULL;
    msg->header.entitySize = 0;
    msg->reader = DefaultReadEntity;
    msg->writer = DefaultWriteEntity;
    return msg;
}

KVMsg* NewEmptyMsg(void) {
    KVMsg* msg = palloc0(sizeof(KVMsg));
    msg->entity = NULL;
    msg->header.entitySize = 0;
    msg->reader = DefaultReadEntity;
    msg->writer = DefaultWriteEntity;
    return msg;
}

void DefaultWriteEntity(KVChannel* channel, uint64* offset, void* entity,
                        uint64 size) {
    char* buf = channel->shared->buffer;
    uint64 buf_size = KV_CHANNEL_BUFSIZE;
    uint64 pos = *offset;
    uint64 remaining = buf_size - pos;

    Assert(pos >= 0 && pos < buf_size);

    if (size == 0) return;

    if (size <= remaining) {
        memcpy(buf + pos, entity, size);
    } else {
        memcpy(buf + pos, entity, remaining);
        memcpy(buf, (char*)entity + remaining, size - remaining);
    }

    *offset = (pos + size) % buf_size;
}

void DefaultReadEntity(KVChannel* channel, uint64* offset, void* entity,
                       uint64 size) {
    char* buf = channel->shared->buffer;
    uint64 buf_size = KV_CHANNEL_BUFSIZE;
    uint64 pos = *offset;
    uint64 remaining = buf_size - pos;

    Assert(pos >= 0 && pos < buf_size);

    if (size == 0) return;

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

    if (msg == NULL) {
        /* Print msg */
        elog(INFO, "KVMsg: NULL");
        return;
    }

    /* Decode operation */
    switch (msg->header.op) {
        case kv_none:
            op_str = "NONE";
            break;
        case kv_open:
            op_str = "OPEN";
            break;
        case kv_close:
            op_str = "CLOSE";
            break;
        case kv_count:
            op_str = "COUNT";
            break;
        case kv_put:
            op_str = "PUT";
            break;
        case kv_get:
            op_str = "GET";
            break;
        case kv_delete:
            op_str = "DELETE";
            break;
        case kv_load:
            op_str = "LOAD";
            break;
        case kv_batch_read:
            op_str = "BATCH_READ";
            break;
        case kv_cursor_delete:
            op_str = "CURSOR_DELETE";
            break;
        case kv_start:
            op_str = "START";
            break;
        case kv_stop:
            op_str = "STOP";
            break;
        case kv_range:
            op_str = "RANGE";
            break;
        default:
            op_str = "UNKNOWN_OP";
            elog(ERROR, "PrintKVMsg: unknown op %d", msg->header.op);
            break;
    }

    /* Decode status */
    switch (msg->header.status) {
        case kv_status_none:
            status_str = "NONE";
            break;
        case kv_status_ok:
            status_str = "OK";
            break;
        case kv_status_error:
            status_str = "ERROR";
            break;
        case kv_status_failed:
            status_str = "FAILED";
            break;
        default:
            status_str = "UNKNOWN_STATUS";
            break;
    }

    /* Print header */
    elog(INFO, "KVMsg: op=%s relId=%u status=%s respChannel=%u entitySize=%lu",
         op_str, msg->header.relId, status_str, msg->header.respChannel,
         msg->header.entitySize);

    /* Print entity if applicable */
    if (msg->entity != NULL && msg->header.entitySize > 0) {
        char preview[256];
        uint64 max_print = msg->header.entitySize;

        if (max_print >= sizeof(preview)) max_print = sizeof(preview) - 1;

        memcpy(preview, msg->entity, max_print);

        for (uint64 i = 0; i < max_print; ++i) {
            char c = preview[i];
            if (c >= 32 && c <= 126)  // Printable ASCII range
                preview[i] = c;
            else
                preview[i] = '#';
        }

        preview[max_print] = '\0';

        elog(INFO, "KVMsg entity preview: \"%s\" (truncated if larger)",
             preview);
    } else {
        elog(INFO, "KVMsg entity: NULL");
    }
}

bool KVChannelPushMsg(KVChannel* channel, KVMsg* msg, long timeout_ms) {
    uint64 header_size = sizeof(KVMsgHeader);
    uint64 entity_size = msg->header.entitySize;
    uint64 total_size = header_size + entity_size;
    char temp[MSG_SIZE] = {0};
    uint64 offset = 0;

    if (total_size > MSG_SIZE) {
        elog(ERROR, "KVChannelPushMsg: total message too large (%lu bytes)",
             total_size);
        return false;
    }

    Assert(total_size <= MSG_SIZE);
    memcpy(temp + offset, &msg->header, header_size);
    offset += header_size;
    if (entity_size > 0 && msg->entity != NULL) {
        memcpy(temp + offset, msg->entity, entity_size);
        offset += entity_size;
    }

    return KVChannelPush(channel, temp, total_size, timeout_ms);
}

KVMsg* KVChannelPopMsg(KVChannel* channel, long timeout_ms) {
    instr_time start_time, cur_time;
    long cur_timeout = timeout_ms;
    KVMsg* msg = NewEmptyMsg();
    bool block = timeout_ms != 0;

    if (block && timeout_ms > 0) INSTR_TIME_SET_CURRENT(start_time);
    // NRAM_TEST_INFO("Calling KVChannelPopMsg from proc %d", MyProcPid);

    while (pg_atomic_read_u32(&channel->shared->is_running)) {
        LWLockAcquire(&channel->shared->lock, LW_EXCLUSIVE);

        if (UnsafeKVChannelPop(channel, &msg->header, sizeof(KVMsgHeader))) {
            NRAM_TEST_INFO("[Pop] Got header to be reply to %u", msg->header.respChannel);
            msg->entity = palloc(msg->header.entitySize);

            if (!UnsafeKVChannelPop(channel, msg->entity,
                                    msg->header.entitySize)) {
                elog(ERROR,
                     "Corrupted message: the remaining message is smaller than "
                     "entitySize %lu",
                     msg->header.entitySize);
            }

            LWLockRelease(&channel->shared->lock);
            return msg;
        }

        if (!block) {
            LWLockRelease(&channel->shared->lock);
            pfree(msg);
            return NULL;
        }

        /* Timeout logic */
        if (timeout_ms > 0) {
            INSTR_TIME_SET_CURRENT(cur_time);
            INSTR_TIME_SUBTRACT(cur_time, start_time);
            cur_timeout = timeout_ms - (long)INSTR_TIME_GET_MILLISEC(cur_time);

            if (cur_timeout <= 0) {
                LWLockRelease(&channel->shared->lock);
                NRAM_TEST_INFO("KVChannelPopMsg encounters timeout");
                pfree(msg);
                return NULL;
            }
        } else
            cur_timeout = -1;  // No timeout, infinite block

        ConditionVariablePrepareToSleep(&channel->shared->cv);
        LWLockRelease(&channel->shared->lock);

        if (!PostmasterIsAlive())
            proc_exit(0);

        if (cur_timeout > 0)
            ConditionVariableTimedSleep(&channel->shared->cv, cur_timeout, WAIT_EVENT_KV_CHANNEL);
        else
            ConditionVariableSleep(&channel->shared->cv, WAIT_EVENT_KV_CHANNEL);
        ConditionVariableCancelSleep();
    }

    pfree(msg);
    return NULL;
}


void TerminateChannel(KVChannel* channel) {
    NRAM_TEST_INFO("Terminating the channel %s", channel->name);
    pg_atomic_write_u32(&channel->shared->is_running, 0);
    ConditionVariableBroadcast(&channel->shared->cv);
    SetLatch(MyLatch);
}
