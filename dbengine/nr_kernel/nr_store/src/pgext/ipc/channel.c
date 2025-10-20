#include "pgext/ipc/channel.h"

#include <miscadmin.h>
#include <portability/instr_time.h>
#include <storage/shmem.h>
#include <storage/pmsignal.h>
#include <storage/ipc.h>
#include <utils/wait_event.h>
#include <storage/latch.h>


static int  NSChannelTrancheId = 0;

int GetNSChannelTrancheId(void) {
    return NSChannelTrancheId;
}

void NSChannelRegisterTranche(void) {
    if (NSChannelTrancheId == 0) {
        NSChannelTrancheId = LWLockNewTrancheId();
        LWLockRegisterTranche(NSChannelTrancheId, "NeurStoreChannel");
    }
}

NSChannel* NSChannelInit(const char* name, bool create) {
    bool exist;
    NSChannel* channel = (NSChannel*) ShmemInitStruct(name, sizeof(NSChannel), &exist);

    if (create && !exist) {
        memset(channel, 0, sizeof(NSChannel));
        pg_atomic_init_u32(&channel->is_active, 1);
        strlcpy(channel->name, name, NAMEDATALEN);
        LWLockInitialize(&channel->lock, GetNSChannelTrancheId());
        ConditionVariableInit(&channel->cv);
    }
    return channel;
}

bool NSChannelPush(NSChannel *channel, const NSTaskMsg *message) {
    uint64 used, available, start_pos, end_pos, middle_pos;
    instr_time start_time, current_time;
    long time_left;

    Size msg_size = NSTASK_HDRSZ + message->payload_size;
    if (msg_size > NS_MSG_SIZE) {
        elog(WARNING, "NSChannelPush: message too large (%zu bytes)", msg_size);
        return false;
    }

    INSTR_TIME_SET_CURRENT(start_time);

    while (pg_atomic_read_u32(&channel->is_active)) {
        LWLockAcquire(&channel->lock, LW_EXCLUSIVE);
        used = (channel->tail + NS_CHANNEL_BUFSIZE - channel->head) % NS_CHANNEL_BUFSIZE;
        available = NS_CHANNEL_BUFSIZE - used - 1;

        if (msg_size <= available) {
            // push the message
            start_pos = channel->tail;
            end_pos = (start_pos + msg_size) % NS_CHANNEL_BUFSIZE;

            if (start_pos + msg_size <= NS_CHANNEL_BUFSIZE) {
                memcpy(channel->buffer + start_pos, message, msg_size);
            } else {
                middle_pos = NS_CHANNEL_BUFSIZE - start_pos;
                memcpy(channel->buffer + start_pos, message, middle_pos);
                memcpy(channel->buffer, ((char*)message) + middle_pos, msg_size - middle_pos);
            }
            channel->tail = end_pos;
            LWLockRelease(&channel->lock);
            ConditionVariableSignal(&channel->cv);
            return true;
        }

        // blocking, check for timeout
        INSTR_TIME_SET_CURRENT(current_time);
        INSTR_TIME_SUBTRACT(current_time, start_time);
        long elapsed_time = INSTR_TIME_GET_MILLISEC(current_time);
        if (elapsed_time >= NS_CHANNEL_TIMEOUT) {
            LWLockRelease(&channel->lock);
            // elog(WARNING, "NSChannelPush: timeout after %ld ms", elapsed_time);
            return false;
        }
        time_left = NS_CHANNEL_TIMEOUT - elapsed_time;

        // sleep until notified
        ConditionVariablePrepareToSleep(&channel->cv);
        LWLockRelease(&channel->lock);

        if (!PostmasterIsAlive()) {
            // Postmaster is dead, exit the process
            proc_exit(1);
        }
        ConditionVariableTimedSleep(&channel->cv, time_left, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();
    }
    return false;
}

NSTaskMsg* NSChannelPop(NSChannel *channel) {
    uint64 used, start_pos, end_pos, middle_pos;
    instr_time start_time, current_time;
    long timeout_ms = NS_CHANNEL_TIMEOUT;
    const Size header_size = NSTASK_HDRSZ;

    INSTR_TIME_SET_CURRENT(start_time);

    while (pg_atomic_read_u32(&channel->is_active)) {
        LWLockAcquire(&channel->lock, LW_EXCLUSIVE);

        used = (channel->tail + NS_CHANNEL_BUFSIZE - channel->head) % NS_CHANNEL_BUFSIZE;

        if (used >= header_size) {
            // message is available
            char header_buf[header_size];
            start_pos = channel->head;

            if (start_pos + header_size <= NS_CHANNEL_BUFSIZE)
                // copy header directly
                memcpy(header_buf, channel->buffer + start_pos, header_size);
            else {
                middle_pos = NS_CHANNEL_BUFSIZE - start_pos;
                memcpy(header_buf, channel->buffer + start_pos, middle_pos);
                memcpy(header_buf + middle_pos, channel->buffer, header_size - middle_pos);
            }

            // parse header
            uint32 *payload_size = (uint32*)(header_buf + sizeof(NSTaskType));
            Size total_size = header_size + *payload_size;

            // read the entire message
            if (used >= total_size) {
                NSTaskMsg *msg = (NSTaskMsg*) palloc0(total_size);
                end_pos = (start_pos + total_size) % NS_CHANNEL_BUFSIZE;

                if (start_pos + total_size <= NS_CHANNEL_BUFSIZE)
                    // copy the entire message
                    memcpy(msg, channel->buffer + start_pos, total_size);
                else {
                    middle_pos = NS_CHANNEL_BUFSIZE - start_pos;
                    memcpy(msg, channel->buffer + start_pos, middle_pos);
                    memcpy(((char*)msg) + middle_pos, channel->buffer, total_size - middle_pos);
                }
                channel->head = end_pos;
                LWLockRelease(&channel->lock);
                ConditionVariableSignal(&channel->cv);
                return msg;
            } else {
                LWLockRelease(&channel->lock);
                elog(WARNING, "NSChannelPopMsg: header ok, but message incomplete (used=%lu, need=%zu)", used, total_size);
                return NULL;
            }
        }

        INSTR_TIME_SET_CURRENT(current_time);
        INSTR_TIME_SUBTRACT(current_time, start_time);
        long elapsed = INSTR_TIME_GET_MILLISEC(current_time);
        if (elapsed >= timeout_ms) {
            LWLockRelease(&channel->lock);
            // elog(WARNING, "NSChannelPop: timeout after %ld ms", elapsed);
            return NULL;
        }
        long cur_timeout = timeout_ms - elapsed;
        ConditionVariablePrepareToSleep(&channel->cv);
        LWLockRelease(&channel->lock);
        if (!PostmasterIsAlive())
            proc_exit(1);
        ConditionVariableTimedSleep(&channel->cv, cur_timeout, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();
    }
    return NULL;
}

void
NSChannelDestroy(NSChannel *channel) {
    if (!channel)
        return;
    pg_atomic_write_u32(&channel->is_active, 0);
    ConditionVariableBroadcast(&channel->cv);
}

NSTaskMsg* NewEmptyTask() {
    NSTaskMsg *task = (NSTaskMsg *) palloc0(sizeof(NSTaskMsg));
    task->type = NS_TASK_NONE;
    task->payload_size = 0;
    task->resp_channel = 0;
    return task;
}

NSTaskMsg* NewTask(NSTaskType type, uint32 payload_size, uint32 resp_channel, const void *payload) {
    const Size total_size = NSTASK_HDRSZ + payload_size;
    if (total_size > NS_MSG_SIZE) {
        elog(WARNING, "NewTask: payload too large (%u bytes)", payload_size);
        return NULL;
    }

    NSTaskMsg *task = (NSTaskMsg *) palloc0(total_size);
    task->type = type;
    task->payload_size = payload_size;
    task->resp_channel = resp_channel;
    if (payload_size > 0 && payload != NULL)
        memcpy(task->payload, payload, payload_size);
    return task;
}

NSTaskMsg* NewOkResponse() {
    NSTaskMsg *resp = (NSTaskMsg *) palloc0(NSTASK_HDRSZ);
    resp->type = NS_TASK_OK;
    resp->payload_size = 0;
    resp->resp_channel = 0;
    return resp;
}

NSTaskMsg* NewErrorResponse(const char *error_msg) {
    Size len = strlen(error_msg) + 1;
    Size total_size = NSTASK_HDRSZ + len;

    if (total_size > NS_MSG_SIZE) {
        elog(WARNING, "NewErrorResponse: error message too long");
        len = NS_MSG_SIZE - NSTASK_HDRSZ;
        total_size = NSTASK_HDRSZ + len;
    }

    NSTaskMsg *resp = (NSTaskMsg *) palloc0(total_size);
    resp->type = NS_TASK_ERROR;
    resp->payload_size = len;
    resp->resp_channel = 0;

    memcpy(resp->payload, error_msg, len);
    return resp;
}
