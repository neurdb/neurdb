#include "nram_storage/rocks_service.h"
#include "nram_storage/thread.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"

static BackgroundWorker worker;
static KVChannel *channel = NULL;
static volatile sig_atomic_t rocks_service_running = false;

void run_rocks(int num_threads) {
    static ThreadPool pool;
    channel = KVChannelInit(ROCKSDB_CHANNEL, true);

    NRAM_TEST_INFO("[Rocks] Service started with %d threads (pid=%d)", num_threads, MyProcPid);
    Assert(!rocks_service_running);

    threadpool_init(&pool, num_threads);
    rocks_service_running = true;

    while (rocks_service_running) {
        KVMsg* msg = KVChannelPopMsg(channel, true);
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

        threadpool_add_task(&pool, process_request, msg);
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

    NRAM_TEST_INFO("[Rocks] Service started (pid=%d)", MyProcPid);

    Assert(!rocks_service_running);
    rocks_service_running = true;

    while (rocks_service_running) {
        KVMsg *msg = KVChannelPopMsg(channel, true);
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
        CHECK_FOR_INTERRUPTS();
    }

    KVChannelDestroy(channel);
    rocksengine_destroy(GetCurrentEngine());

    NRAM_TEST_INFO("[Rocks] Service terminated (pid=%d)", MyProcPid);
    proc_exit(0);
}


void *process_request(void *arg) {
    KVMsg *msg = (KVMsg *)arg, *resp = NULL;
    KVChannel *resp_chan;
    char chan_name[64];
    
    snprintf(chan_name, sizeof(chan_name), "kv_resp_%u", msg->header.respChannel);
    resp_chan = KVChannelInit(chan_name, false);

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
        default:
            NRAM_TEST_INFO("[Rocks] Unknown op=%d", msg->header.op);
            break;
    }

    if (resp != NULL) {
        NRAM_TEST_INFO("[Rocks] Sending response op=%d, size=%lu", resp->header.op, resp->header.entitySize);
        PrintKVMsg(resp);
        KVChannelPushMsg(resp_chan, resp, true);

        if (resp->entity)
            pfree(resp->entity);
        pfree(resp);
    }

    if (msg->entity)
        pfree(msg->entity);
    pfree(msg);

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
    int result_count;
    char *write_ptr;
    Size total_len;
    KVMsg *resp;
    NRAM_INFO();

    Assert(msg->entity != NULL && msg->header.entitySize > 0);

    /* Deserialize start_key and end_key */
    memcpy(&key_len_1, msg->entity, sizeof(Size));
    start_key = tkey_deserialize(msg->entity + sizeof(Size), key_len_1);

    memcpy(&key_len_2, msg->entity + sizeof(Size) + key_len_1, sizeof(Size));
    end_key = tkey_deserialize(msg->entity + sizeof(Size) + key_len_1 + sizeof(Size), key_len_2);

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
    NRAM_TEST_INFO("Total len = %lu, data len = %lu", total_len, result_count);

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

    strncpy(worker.bgw_name, "Rocks Service", BGW_MAXLEN - 1);
    worker.bgw_name[BGW_MAXLEN - 1] = '\0';
    strncpy(worker.bgw_type, "CustomStorageWorker", BGW_MAXLEN - 1);
    worker.bgw_type[BGW_MAXLEN - 1] = '\0';

    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_ConsistentState;
    worker.bgw_restart_time = 1;
    worker.bgw_main_arg = UInt32GetDatum(0);

    snprintf(worker.bgw_library_name, BGW_MAXLEN, "nram");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "rocks_service_main");

    worker.bgw_notify_pid = 0;

    RegisterBackgroundWorker(&worker);
}

void nram_rocks_service_terminate(void) {
    terminate_rocks(0);
}
