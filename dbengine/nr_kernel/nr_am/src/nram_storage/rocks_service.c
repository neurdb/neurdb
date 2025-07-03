#include "nram_storage/rocks_service.h"
#include "nram_storage/thread.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/pmsignal.h"

RocksEngine *GlobalRocksEngine = NULL;
static BackgroundWorker worker;
static KVChannel *channel = NULL;

void run_rocks(int num_threads) {
    static ThreadPool pool;
    channel = KVChannelInit(ROCKSDB_CHANNEL, true);

    NRAM_TEST_INFO("[Rocks] Service started with %d threads (pid=%d)", num_threads, MyProcPid);
    Assert(!rocks_service_running);

    threadpool_init(&pool, num_threads);
    GlobalRocksEngine = rocksengine_open();
    rocks_service_running = true;

    while (rocks_service_running && PostmasterIsAlive()) {
        KVMsg *msg = (KVMsg *)malloc(sizeof(KVMsg));
        bool ok = KVChannelPopMsg(channel, msg, true);
        if (!ok) {
            free(msg);
            CHECK_FOR_INTERRUPTS();
            continue;
        }

        NRAM_TEST_INFO("[Rocks] Received msg op=%d, respChan=%u, size=%lu (pid=%d)", 
                        msg->header.op, msg->header.respChannel, msg->header.entitySize, MyProcPid);

        if (msg->header.op == kv_close) {
            NRAM_TEST_INFO("[Rocks] Received kv_close, shutting down (pid=%d)", MyProcPid);
            free(msg->entity);
            free(msg);
            break;
        }

        threadpool_add_task(&pool, process_request, msg);
        CHECK_FOR_INTERRUPTS();
    }

    NRAM_TEST_INFO("fin: next step destroy thread pool");
    threadpool_destroy(&pool);
    NRAM_TEST_INFO("fin: next step destroy channel");
    KVChannelDestroy(channel);
    NRAM_TEST_INFO("fin: next step destroy engine");
    rocksengine_destroy(&GlobalRocksEngine->engine);

    NRAM_TEST_INFO("[Rocks] Service terminated (pid=%d)", MyProcPid);
    proc_exit(0);
}

void run_rocks_no_thread(void) {
    NRAM_INFO();
    channel = KVChannelInit(ROCKSDB_CHANNEL, true);

    NRAM_TEST_INFO("[Rocks] Service started (pid=%d)", MyProcPid);
    Assert(!rocks_service_running);

    GlobalRocksEngine = rocksengine_open();
    rocks_service_running = true;

    while (rocks_service_running && PostmasterIsAlive()) {
        KVMsg *msg = (KVMsg *)malloc(sizeof(KVMsg));
        bool ok = KVChannelPopMsg(channel, msg, true);
        if (!ok) {
            free(msg);
            CHECK_FOR_INTERRUPTS();
            continue;
        }

        NRAM_TEST_INFO("[Rocks] Received msg op=%d, respChan=%u, size=%lu (pid=%d)", 
                        msg->header.op, msg->header.respChannel, msg->header.entitySize, MyProcPid);

        if (msg->header.op == kv_close) {
            NRAM_TEST_INFO("[Rocks] Received kv_close, shutting down (pid=%d)", MyProcPid);
            free(msg->entity);
            free(msg);
            break;
        }

        process_request(msg);
        CHECK_FOR_INTERRUPTS();
    }

    NRAM_TEST_INFO("fin: next step destroy channel");
    KVChannelDestroy(channel);
    NRAM_TEST_INFO("fin: next step destroy engine");
    rocksengine_destroy(&GlobalRocksEngine->engine);

    NRAM_TEST_INFO("[Rocks] Service terminated (pid=%d)", MyProcPid);
    proc_exit(0);
}

static void terminate_rocks(SIGNAL_ARGS) {
    int save_errno = errno;
    rocks_service_running = false;
    if (channel)
        ConditionVariableBroadcast(&channel->shared->cv);
    errno = save_errno;
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
            free(resp->entity);
        free(resp);
    }

    if (resp_chan)
        KVChannelDestroy(resp_chan);

    if (msg->entity)
        free(msg->entity);
    free(msg);

    return NULL;
}


KVMsg *handle_kv_get(KVMsg *msg) {
    Size key_len = msg->header.entitySize;
    NRAMKey key = tkey_deserialize((char *)msg->entity, key_len);
    NRAMValue value = rocksengine_get(&GlobalRocksEngine->engine, key);

    KVMsg *resp = malloc(sizeof(KVMsg));
    Size val_len;

    NRAM_TEST_INFO("[Rocks] handle_kv_get, key_len=%lu, tableOid=%u", key_len, key->tableOid);

    *resp = NewStatusMsg(kv_status_ok, msg->header.respChannel);
    resp->header.op = kv_get;

    Assert(key_len > 0 && msg->entity != NULL);
    resp->entity = tvalue_serialize(value, &val_len);
    resp->header.entitySize = val_len;
    resp->header.relId = key->tableOid;

    free(key);
    if (value)
        free(value);

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

    rocksengine_put(&GlobalRocksEngine->engine, key, value);

    resp = malloc(sizeof(KVMsg));
    *resp = NewStatusMsg(kv_status_ok, msg->header.respChannel);
    resp->header.op = kv_put;
    resp->header.relId = key->tableOid;

    free(key);
    free(value);

    return resp;
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
}
