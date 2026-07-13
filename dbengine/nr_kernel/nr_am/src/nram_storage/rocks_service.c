#include "nram_storage/rocks_service.h"
#include "nram_storage/thread.h"
#include "nram_storage/indexengine.h"
#include "nrindex_access/nrindex_kv.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"

static BackgroundWorker worker;
static KVChannel *channel = NULL;
static volatile sig_atomic_t rocks_service_running = false;
static ResultQueue result_queue;
static IndexEngine *index_engine = NULL;

/* ------------------------------------------------------------------------
 * Response thread-safe queue:
 *  we add this because the channel can only be used in process context.
 * ------------------------------------------------------------------------
 */

void ResultQueueInit(ResultQueue *q) {
    NRAM_INFO();
    pthread_mutex_init(&q->lock, NULL);
    q->head = q->tail = NULL;
    q->shutdown = false;
}

void ResultQueueDestroy(ResultQueue *q) {
    ResultNode *curr, *tmp;
    NRAM_INFO();
    pthread_mutex_lock(&q->lock);
    q->shutdown = true;
    pthread_mutex_unlock(&q->lock);

    curr = q->head;
    while (curr) {
        tmp = curr;
        curr = curr->next;
        if (tmp->msg) {
            if (tmp->msg->entity)
                pfree(tmp->msg->entity);
            pfree(tmp->msg);
        }
        free(tmp);
    }

    pthread_mutex_destroy(&q->lock);
}

void ResultQueuePush(ResultQueue *q, KVMsg *msg) {
    ResultNode *node = malloc(sizeof(ResultNode));
    NRAM_INFO();
    node->msg = msg;
    node->next = NULL;

    pthread_mutex_lock(&q->lock);
    if (q->shutdown) {
        pthread_mutex_unlock(&q->lock);
        return;
    }

    if (q->tail)
        q->tail->next = node;
    else
        q->head = node;
    q->tail = node;
    pthread_mutex_unlock(&q->lock);
}

KVMsg *ResultQueuePop(ResultQueue *q) {
    ResultNode *node;
    KVMsg *msg;
    NRAM_INFO();
    pthread_mutex_lock(&q->lock);
    node = q->head;
    if (q->shutdown || node == NULL) {
        pthread_mutex_unlock(&q->lock);
        return NULL;
    }

    q->head = node->next;
    if (q->head == NULL) q->tail = NULL;
    pthread_mutex_unlock(&q->lock);

    msg = node->msg;
    free(node);
    return msg;
}

void PrintResultQueue(ResultQueue *q) {
    pthread_mutex_lock(&q->lock);
    elog(INFO, "<ResultQueue: %p (head:%p, tail:%p, status:%s) --------------- >",
        q, q->head, q->tail, q->shutdown? "shutdown":"alive");
    if (q->shutdown) {
        pthread_mutex_unlock(&q->lock);
        return;
    }
    for (ResultNode* it = q->head; it; it = it->next) {
        PrintKVMsg(it->msg);
        elog(INFO, "--------------- ");
    }
    elog(INFO, "<ResultQueue: %p ---------------> ", q);
    pthread_mutex_unlock(&q->lock);
}

bool ResultQueueIsEmpty(ResultQueue *q) {
    return q == NULL || (q->head == NULL) || q->shutdown;
}


static void ResultQueueClear(ResultQueue *q) {
    // NRAM_INFO();
    // PrintResultQueue(q);
    KVChannel *resp_chan;
    FILE *f;
    bool ok;

    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] ResultQueueClear: ENTER, isEmpty=%d\n", ResultQueueIsEmpty(q));
        fflush(f);
        fclose(f);
    }

    while (!ResultQueueIsEmpty(q)) {
        KVMsg *resp = ResultQueuePop(q);
        if (resp) {
            char chan_name[64];
            snprintf(chan_name, sizeof(chan_name), "kv_resp_%u", resp->header.respChannel);

            f = fopen("/tmp/nrindex_debug.log", "a");
            if (f) {
                fprintf(f, "[Service] ResultQueueClear: sending to channel '%s', op=%d, entitySize=%lu\n",
                        chan_name, resp->header.op, resp->header.entitySize);
                fflush(f);
                fclose(f);
            }

            resp_chan = KVChannelInit(chan_name, false);
            ok = KVChannelPushMsg(resp_chan, resp, -1);

            f = fopen("/tmp/nrindex_debug.log", "a");
            if (f) {
                fprintf(f, "[Service] ResultQueueClear: KVChannelPushMsg returned %d\n", ok);
                fflush(f);
                fclose(f);
            }

            if (resp->entity)
                pfree(resp->entity);
            pfree(resp);
            pfree(resp_chan);
        }
    }

    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] ResultQueueClear: EXIT\n");
        fflush(f);
        fclose(f);
    }
}


/* ------------------------------------------------------------------------
 * Main funcs
 * ------------------------------------------------------------------------
 */

void run_rocks(int num_threads) {
    static ThreadPool pool;
    channel = KVChannelInit(ROCKSDB_CHANNEL, true);
    ResultQueueInit(&result_queue);

    NRAM_TEST_INFO("[Rocks] Service started with %d threads (pid=%d)", num_threads, MyProcPid);
    Assert(!rocks_service_running);

    threadpool_init(&pool, num_threads);
    rocks_service_running = true;

    while (rocks_service_running) {
        KVMsg* msg = KVChannelPopMsg(channel, ROCKSDB_CHANNEL_DEFAULT_TIMEOUT);
        if (msg == NULL) {
            CHECK_FOR_INTERRUPTS();
            ResultQueueClear(&result_queue);
            continue;
        }

        NRAM_TEST_INFO("[Rocks] Received msg op=%d, respChan=%u, size=%lu (pid=%d)",
                        msg->header.op, msg->header.respChannel, msg->header.entitySize, MyProcPid);

        if (msg->header.op == kv_close) {
            NRAM_TEST_INFO("[Rocks] Received kv_close, shutting down (pid=%d)", MyProcPid);
            pfree(msg->entity);
            pfree(msg);
            break;
        }

        threadpool_add_task(&pool, process_request, msg);

        ResultQueueClear(&result_queue);
        CHECK_FOR_INTERRUPTS();
    }

    threadpool_destroy(&pool);
    KVChannelDestroy(channel);
    rocksengine_destroy(GetCurrentEngine());

    NRAM_TEST_INFO("[Rocks] Service terminated (pid=%d)", MyProcPid);
    proc_exit(0);
}

void run_rocks_no_thread(void) {
    NRAM_INFO();
    channel = KVChannelInit(ROCKSDB_CHANNEL, true);
    ResultQueueInit(&result_queue);

    NRAM_TEST_INFO("[Rocks] Service started (pid=%d)", MyProcPid);

    Assert(!rocks_service_running);
    rocks_service_running = true;

    while (rocks_service_running) {
        KVMsg *msg = KVChannelPopMsg(channel, ROCKSDB_CHANNEL_DEFAULT_TIMEOUT);
        if (msg == NULL) {
            CHECK_FOR_INTERRUPTS();
            continue;
        }

        NRAM_TEST_INFO("[Rocks] Received msg op=%d, respChan=%u, size=%lu (pid=%d)",
                        msg->header.op, msg->header.respChannel, msg->header.entitySize, MyProcPid);

        if (msg->header.op == kv_close) {
            NRAM_TEST_INFO("[Rocks] Received kv_close, shutting down (pid=%d)", MyProcPid);
            pfree(msg->entity);
            pfree(msg);
            break;
        }

        process_request(msg);
        ResultQueueClear(&result_queue);
        CHECK_FOR_INTERRUPTS();
    }

    KVChannelDestroy(channel);
    rocksengine_destroy(GetCurrentEngine());

    NRAM_TEST_INFO("[Rocks] Service terminated (pid=%d)", MyProcPid);
    proc_exit(0);
}

void *process_request(void *arg) {
    KVMsg *msg = (KVMsg *)arg, *resp = NULL;
    char chan_name[64];

    snprintf(chan_name, sizeof(chan_name), "kv_resp_%u", msg->header.respChannel);
    NRAM_TEST_INFO("[Rocks] Processing msg op=%d, respChan=%u (pid=%d)",
        msg->header.op, msg->header.respChannel, MyProcPid);

    switch (msg->header.op) {
        case kv_open:
            NRAM_TEST_INFO("[Rocks] kv_open ignored");
            break;
        case kv_close:
            NRAM_TEST_INFO("[Rocks] kv_close ignored in worker");
            break;
        case kv_get:
            resp = handle_kv_get(msg);
            break;
        case kv_put:
            resp = handle_kv_put(msg);
            break;
        case kv_range:
            resp = handle_kv_range_scan(msg);
            break;
        case kv_delete:
            NRAM_TEST_INFO("[Rocks] kv_delete not implemented");
            break;
        case kv_index_get:
            resp = handle_kv_index_get(msg);
            break;
        case kv_index_put:
            resp = handle_kv_index_put(msg);
            break;
        case kv_index_delete:
            resp = handle_kv_index_delete(msg);
            break;
        case kv_index_range_scan:
            resp = handle_kv_index_range_scan(msg);
            break;
        case kv_index_bulk_load:
            resp = handle_kv_index_bulk_load(msg);
            break;
        default:
            NRAM_TEST_INFO("[Rocks] Unknown op=%d", msg->header.op);
            break;
    }

    if (resp != NULL) {
        FILE *f = fopen("/tmp/nrindex_debug.log", "a");
        if (f) {
            fprintf(f, "[Service] process_request: pushing response to queue, op=%d, entitySize=%lu\n",
                    resp->header.op, resp->header.entitySize);
            fflush(f);
            fclose(f);
        }
        ResultQueuePush(&result_queue, resp);  // push to result queue
        f = fopen("/tmp/nrindex_debug.log", "a");
        if (f) {
            fprintf(f, "[Service] process_request: pushed to queue, isEmpty=%d\n",
                    ResultQueueIsEmpty(&result_queue));
            fflush(f);
            fclose(f);
        }
        PrintResultQueue(&result_queue);
    } else {
        if (msg->entity)
            pfree(msg->entity);
        pfree(msg);
    }

    return NULL;
}


KVMsg *handle_kv_get(KVMsg *msg) {
    Size key_len = msg->header.entitySize;
    NRAMKey key = tkey_deserialize((char *)msg->entity, key_len);
    NRAMValue value = rocksengine_get(GetCurrentEngine(), key);
    KVMsg *resp = NewMsg(kv_get, key->tableOid, kv_status_ok, msg->header.respChannel);
    Size val_len;

    NRAM_TEST_INFO("[Rocks] handle_kv_get, key_len=%lu, tableOid=%u", key_len, key->tableOid);
    Assert(key_len > 0 && msg->entity != NULL);
    resp->entity = tvalue_serialize(value, &val_len);
    resp->header.entitySize = val_len;
    resp->header.relId = key->tableOid;

    pfree(key);
    if (value)
        pfree(value);

    return resp;
}


KVMsg *handle_kv_put(KVMsg *msg) {
    Size total_len = msg->header.entitySize, key_len, value_len;
    char *buf = (char *)msg->entity;
    NRAMKey key;
    NRAMValue value;
    KVMsg *resp;

    if (total_len < sizeof(Size)) {
        NRAM_TEST_INFO("[Rocks] Invalid kv_put message: size too small");
        return NULL;
    }

    memcpy(&key_len, buf, sizeof(Size));
    buf += sizeof(Size);

    if (total_len < sizeof(Size) + key_len) {
        NRAM_TEST_INFO("[Rocks] Invalid kv_put message: key length mismatch");
        return NULL;
    }

    key = tkey_deserialize(buf, key_len);
    buf += key_len;

    value_len = total_len - key_len - sizeof(Size);
    value = tvalue_deserialize(buf, value_len);

    NRAM_TEST_INFO("[Rocks] handle_kv_put, key_len=%lu, val_len=%lu, tableOid=%u", key_len, value_len, key->tableOid);

    rocksengine_put(GetCurrentEngine(), key, value);

    resp = NewMsg(kv_put, msg->header.relId, kv_status_ok, msg->header.respChannel);
    resp->header.op = kv_put;
    resp->header.relId = key->tableOid;

    pfree(key);
    pfree(value);

    return resp;
}

KVMsg *handle_kv_range_scan(KVMsg *msg) {
    Size key_len_1, key_len_2;
    NRAMKey start_key, end_key, *keys;
    NRAMValue *results;
    uint32_t result_count;
    char *write_ptr;
    Size total_len;
    KVMsg *resp;
    NRAM_INFO();

    Assert(msg->entity != NULL && msg->header.entitySize > 0);

    /* Deserialize start_key and end_key */
    memcpy(&key_len_1, msg->entity, sizeof(Size));
    start_key = tkey_deserialize((char *)msg->entity + sizeof(Size), key_len_1);

    memcpy(&key_len_2, (char *)msg->entity + sizeof(Size) + key_len_1, sizeof(Size));
    end_key = tkey_deserialize((char *)msg->entity + sizeof(Size) + key_len_1 + sizeof(Size), key_len_2);

    NRAM_TEST_INFO("[Rocks] handle_kv_range_scan, [table %u - %lu, table %u - %lu)",
                   start_key->tableOid, start_key->tid, end_key->tableOid, end_key->tid);

    // Assert(start_key->tableOid == end_key->tableOid);

    /* Query range from RocksDB */
    rocksengine_range_scan(GetCurrentEngine(), start_key, end_key, &result_count, &keys, &results);

    total_len = sizeof(int);  // result_count
    for (int i = 0; i < result_count; i++) {
        Size klen, vlen;
        char *kbuf = tkey_serialize(keys[i], &klen);
        char *vbuf = tvalue_serialize(results[i], &vlen);
        total_len += sizeof(Size) + klen + sizeof(Size) + vlen;
        pfree(kbuf);
        pfree(vbuf);
    }
    NRAM_TEST_INFO("Total len = %lu, data len = %u", total_len, result_count);

    resp = NewMsg(kv_range, start_key->tableOid, kv_status_ok, msg->header.respChannel);
    resp->header.entitySize = total_len;
    resp->entity = palloc(total_len);

    write_ptr = resp->entity;
    memcpy(write_ptr, &result_count, sizeof(int));
    write_ptr += sizeof(int);

    if (result_count) {
        for (int i = 0; i < result_count; i++) {
            Size klen, vlen;
            char *kbuf = tkey_serialize(keys[i], &klen);
            char *vbuf = tvalue_serialize(results[i], &vlen);

            memcpy(write_ptr, &klen, sizeof(Size));
            write_ptr += sizeof(Size);
            memcpy(write_ptr, kbuf, klen);
            write_ptr += klen;

            memcpy(write_ptr, &vlen, sizeof(Size));
            write_ptr += sizeof(Size);
            memcpy(write_ptr, vbuf, vlen);
            write_ptr += vlen;

            pfree(kbuf);
            pfree(vbuf);
            pfree(keys[i]);
            pfree(results[i]);
        }

        pfree(keys);
        pfree(results);
    }
    pfree(start_key);
    pfree(end_key);
    // PrintKVMsg(resp);

    return resp;
}

/* ------------------------------------------------------------------------
 * Index operation handlers
 * ------------------------------------------------------------------------
 */

KVMsg *handle_kv_index_get(KVMsg *msg) {
    Size key_len = msg->header.entitySize;
    NRIndexKey ikey = nrindex_key_deserialize((char *)msg->entity, key_len);
    NRIndexValue ivalue = indexengine_get(index_engine, ikey);
    KVMsg *resp = NewMsg(kv_index_get, ikey->indexOid, kv_status_ok, msg->header.respChannel);
    Size val_len;

    NRAM_TEST_INFO("[IndexEngine] handle_kv_index_get, key_len=%lu, indexOid=%u", key_len, ikey->indexOid);
    Assert(key_len > 0 && msg->entity != NULL);

    if (ivalue) {
        resp->entity = nrindex_value_serialize(ivalue, &val_len);
        resp->header.entitySize = val_len;
        nrindex_value_free(ivalue);
    } else {
        resp->entity = NULL;
        resp->header.entitySize = 0;
    }
    resp->header.relId = ikey->indexOid;

    nrindex_key_free(ikey);
    return resp;
}

KVMsg *handle_kv_index_put(KVMsg *msg) {
    Size total_len = msg->header.entitySize, key_len, value_len;
    char *buf = (char *)msg->entity;
    NRIndexKey ikey;
    NRIndexValue ivalue;
    KVMsg *resp;

    if (total_len < sizeof(Size)) {
        NRAM_TEST_INFO("[Rocks] Invalid kv_index_put message: size too small");
        return NULL;
    }

    /* Parse key length and value length */
    memcpy(&key_len, buf, sizeof(Size));
    buf += sizeof(Size);

    if (total_len < sizeof(Size) + key_len + sizeof(Size)) {
        NRAM_TEST_INFO("[Rocks] Invalid kv_index_put message: insufficient data");
        return NULL;
    }

    memcpy(&value_len, buf + key_len, sizeof(Size));

    /* Deserialize key and value */
    ikey = nrindex_key_deserialize(buf, key_len);
    buf += key_len + sizeof(Size);
    ivalue = nrindex_value_deserialize(buf, value_len);

    NRAM_TEST_INFO("[IndexEngine] handle_kv_index_put, key_len=%lu, val_len=%lu, indexOid=%u",
                   key_len, value_len, ikey->indexOid);

    indexengine_put(index_engine, ikey, ivalue);

    resp = NewMsg(kv_index_put, msg->header.relId, kv_status_ok, msg->header.respChannel);
    resp->header.op = kv_index_put;
    resp->header.relId = ikey->indexOid;

    nrindex_key_free(ikey);
    nrindex_value_free(ivalue);

    return resp;
}

KVMsg *handle_kv_index_delete(KVMsg *msg) {
    Size key_len = msg->header.entitySize;
    NRIndexKey ikey = nrindex_key_deserialize((char *)msg->entity, key_len);
    KVMsg *resp;

    NRAM_TEST_INFO("[IndexEngine] handle_kv_index_delete, key_len=%lu, indexOid=%u", key_len, ikey->indexOid);

    indexengine_delete(index_engine, ikey);

    resp = NewMsg(kv_index_delete, msg->header.relId, kv_status_ok, msg->header.respChannel);
    resp->header.relId = ikey->indexOid;

    nrindex_key_free(ikey);
    return resp;
}

KVMsg *handle_kv_index_range_scan(KVMsg *msg) {
    Size key_len_1, key_len_2;
    NRIndexKey start_key = NULL, end_key = NULL, *keys = NULL;
    NRIndexValue *results = NULL;
    uint32_t result_count = 0;
    char *write_ptr;
    Size total_len;
    KVMsg *resp;
    char *buf = (char *)msg->entity;
    FILE *f;  /* Debug file handle */

    /* Debug: write to file so we can see if this function is called */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] handle_kv_index_range_scan: ENTER, entitySize=%u\n", msg->header.entitySize);
        fflush(f);
        fclose(f);
    }

    Assert(msg->entity != NULL && msg->header.entitySize > 0);

    /* Parse message: [start_len][start_key_data][end_len][end_key_data]
       If length is 0, the key is NULL */
    memcpy(&key_len_1, buf, sizeof(Size));
    buf += sizeof(Size);
    if (key_len_1 > 0) {
        start_key = nrindex_key_deserialize(buf, key_len_1);
        buf += key_len_1;
    }
    memcpy(&key_len_2, buf, sizeof(Size));
    buf += sizeof(Size);
    if (key_len_2 > 0) {
        end_key = nrindex_key_deserialize(buf, key_len_2);
    }

    elog(LOG, "[IndexEngine] handle_kv_index_range_scan: start_key=%s, end_key=%s",
         start_key ? "SET" : "NULL", end_key ? "SET" : "NULL");
    if (start_key)
        elog(LOG, "[IndexEngine]   start_key: indexOid=%u, key_size=%u",
             start_key->indexOid, start_key->key_size);
    if (end_key)
        elog(LOG, "[IndexEngine]   end_key: indexOid=%u, key_size=%u",
             end_key->indexOid, end_key->key_size);

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] Calling indexengine_range_scan, start_key=%p, end_key=%p\n",
                (void*)start_key, (void*)end_key);
        fflush(f);
        fclose(f);
    }

    /* Query range from IndexEngine */
    indexengine_range_scan(index_engine, start_key, end_key, &result_count, &keys, &results);

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] indexengine_range_scan returned result_count=%u\n", result_count);
        fprintf(f, "[Service] keys=%p, results=%p\n", (void*)keys, (void*)results);
        fflush(f);
        fclose(f);
    }

    /* Validate returned arrays */
    if (result_count > 0 && (keys == NULL || results == NULL)) {
        f = fopen("/tmp/nrindex_debug.log", "a");
        if (f) {
            fprintf(f, "[Service] ERROR: result_count=%u but keys=%p, results=%p\n",
                    result_count, (void*)keys, (void*)results);
            fflush(f);
            fclose(f);
        }
        /* Return error response */
        resp = NewMsg(kv_index_range_scan, msg->header.relId, kv_status_none, msg->header.respChannel);
        resp->entity = NULL;
        resp->header.entitySize = 0;
        if (start_key) nrindex_key_free(start_key);
        if (end_key) nrindex_key_free(end_key);
        return resp;
    }

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] Starting serialization loop for %u results\n", result_count);
        fflush(f);
        fclose(f);
    }

    total_len = sizeof(int);  // result_count
    for (uint32_t i = 0; i < result_count; i++) {
        Size klen, vlen;
        char *kser, *vser;

        /* Debug: log each iteration */
        f = fopen("/tmp/nrindex_debug.log", "a");
        if (f) {
            fprintf(f, "[Service] Serializing item %u: keys[i]=%p, results[i]=%p\n",
                    i, (void*)keys[i], (void*)results[i]);
            fflush(f);
            fclose(f);
        }

        kser = nrindex_key_serialize(keys[i], &klen);
        vser = nrindex_value_serialize(results[i], &vlen);
        total_len += sizeof(Size) + klen + sizeof(Size) + vlen;
        pfree(kser);
        pfree(vser);
    }

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] Serialization loop done, total_len=%zu\n", total_len);
        fflush(f);
        fclose(f);
    }

    resp = NewMsg(kv_index_range_scan, msg->header.relId, kv_status_ok, msg->header.respChannel);
    resp->entity = palloc(total_len);
    resp->header.entitySize = total_len;

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] Created response message, starting final serialization\n");
        fflush(f);
        fclose(f);
    }

    /* Serialize results */
    write_ptr = (char *)resp->entity;
    memcpy(write_ptr, &result_count, sizeof(int));
    write_ptr += sizeof(int);

    for (int i = 0; i < result_count; i++) {
        Size klen, vlen;
        char *kser = nrindex_key_serialize(keys[i], &klen);
        char *vser = nrindex_value_serialize(results[i], &vlen);

        memcpy(write_ptr, &klen, sizeof(Size));
        write_ptr += sizeof(Size);
        memcpy(write_ptr, kser, klen);
        write_ptr += klen;
        memcpy(write_ptr, &vlen, sizeof(Size));
        write_ptr += sizeof(Size);
        memcpy(write_ptr, vser, vlen);
        write_ptr += vlen;

        pfree(kser);
        pfree(vser);
        nrindex_key_free(keys[i]);
        nrindex_value_free(results[i]);
    }

    /* Only free if not NULL (result_count > 0) */
    if (keys)
        pfree(keys);
    if (results)
        pfree(results);
    if (start_key)
        nrindex_key_free(start_key);
    if (end_key)
        nrindex_key_free(end_key);

    /* Debug: write to file */
    f = fopen("/tmp/nrindex_debug.log", "a");
    if (f) {
        fprintf(f, "[Service] handle_kv_index_range_scan: returning response with %u results, resp=%p\n",
                result_count, (void*)resp);
        fflush(f);
        fclose(f);
    }

    return resp;
}

KVMsg *handle_kv_index_bulk_load(KVMsg *msg) {
    char *buf = (char *)msg->entity;
    int count;
    int64 *keys;
    uint64 *values;
    KVMsg *resp;
    Oid indexOid = msg->header.relId;

    NRAM_TEST_INFO("[IndexEngine] handle_kv_index_bulk_load: indexOid=%u, entitySize=%lu",
                   indexOid, msg->header.entitySize);

    /* Parse message: [count (4 bytes)] [keys (count * 8 bytes)] [values (count * 8 bytes)] */
    memcpy(&count, buf, sizeof(int));
    buf += sizeof(int);

    keys = (int64 *)buf;
    buf += count * sizeof(int64);

    values = (uint64 *)buf;

    elog(LOG, "[IndexEngine] handle_kv_index_bulk_load: count=%d", count);

    /* Call indexengine bulk load */
    indexengine_bulk_load(index_engine, indexOid, keys, values, count);

    resp = NewMsg(kv_index_bulk_load, indexOid, kv_status_ok, msg->header.respChannel);
    resp->header.entitySize = 0;
    resp->entity = NULL;

    elog(LOG, "[IndexEngine] handle_kv_index_bulk_load: completed successfully");

    return resp;
}

static void terminate_rocks(SIGNAL_ARGS) {
    int save_errno = errno;
    elog(LOG, "[NRAM] terminate_rocks called, broadcasting CV, setting latch");
    rocks_service_running = false;
    if (channel)
        TerminateChannel(channel);
    errno = save_errno;
}


PGDLLEXPORT void rocks_service_main(Datum arg) {
    uint32_t nthread = DatumGetUInt32(arg);
    NRAM_INFO();
    pqsignal(SIGTERM, terminate_rocks); // automatic shutdown!!
    BackgroundWorkerUnblockSignals();

    if (nthread <= 1) {
        run_rocks_no_thread();
    } else {
        run_rocks(nthread);
    }

    proc_exit(0);
}


void nram_rocks_service_init(void) {
    memset(&worker, 0, sizeof(worker));

    /* Initialize index engine */
    if (index_engine == NULL) {
        index_engine = indexengine_open();
        elog(LOG, "[NRAM] IndexEngine initialized successfully");
    }

    strncpy(worker.bgw_name, "Rocks Service", BGW_MAXLEN - 1);
    worker.bgw_name[BGW_MAXLEN - 1] = '\0';
    strncpy(worker.bgw_type, "CustomStorageWorker", BGW_MAXLEN - 1);
    worker.bgw_type[BGW_MAXLEN - 1] = '\0';

    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_ConsistentState;
    worker.bgw_restart_time = BGW_NEVER_RESTART;
    worker.bgw_main_arg = UInt32GetDatum(1);

    snprintf(worker.bgw_library_name, BGW_MAXLEN, "nram");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "rocks_service_main");

    worker.bgw_notify_pid = 0;

    RegisterBackgroundWorker(&worker);
}

void nram_rocks_service_terminate(void) {
    /* Cleanup index engine */
    if (index_engine != NULL) {
        indexengine_close(index_engine);
        index_engine = NULL;
        elog(LOG, "[NRAM] IndexEngine closed successfully");
    }
    terminate_rocks(0);
}
