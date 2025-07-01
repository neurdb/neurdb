#include "nram_storage/rocks_service.h"
#include "nram_storage/thread.h"

RocksEngine *GlobalRocksEngine = NULL;
static volatile uint32_t rocks_service_status = 0;

void run_rocks(int num_threads) {
    static ThreadPool pool;
    KVChannel *channel = KVChannelInit("rocks_service_channel", true);

    threadpool_init(&pool, num_threads);
    GlobalRocksEngine = rocksengine_open();

    while (rocks_service_status & ROCKS_RUNNING_BIT) {
        KVMsg *msg = (KVMsg *)malloc(sizeof(KVMsg));
        bool ok = KVChannelPopMsg(channel, msg, true);
        if (!ok) {
            free(msg);
            continue;
        }

        if (msg->header.op == kv_close) {
            free(msg->entity);
            free(msg);
            break;
        }

        threadpool_add_task(&pool, process_request, msg);
    }

    threadpool_destroy(&pool);
    KVChannelDestroy(channel);
    rocksengine_destroy(&GlobalRocksEngine->engine);
}


void *process_request(void *arg) {
    KVMsg *msg = (KVMsg *)arg, *resp = NULL;
    KVChannel *resp_chan;
    char chan_name[64];
    
    snprintf(chan_name, sizeof(chan_name), "kv_resp_%u", msg->header.respChannel);
    resp_chan = KVChannelInit(chan_name, false);

    switch (msg->header.op) {
        case kv_open:
            break;
        case kv_close:
            break;
        case kv_get:
            resp = handle_kv_get(msg);
            break;
        case kv_put:
            resp = handle_kv_put(msg);
            break;
        case kv_delete:
            break;
        default:
            break;
    }

    if (resp != NULL) {
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
        elog(WARNING, "Invalid kv_put message: size too small");
        return NULL;
    }

    memcpy(&key_len, buf, sizeof(Size));
    buf += sizeof(Size);

    if (total_len < sizeof(Size) + key_len) {
        elog(WARNING, "Invalid kv_put message: key length mismatch");
        return NULL;
    }

    key = tkey_deserialize(buf, key_len);
    buf += key_len;

    value_len = total_len - key_len - sizeof(Size);
    value = tvalue_deserialize(buf, value_len);

    rocksengine_put(&GlobalRocksEngine->engine, key, value);

    resp = malloc(sizeof(KVMsg));
    *resp = NewStatusMsg(kv_status_ok, msg->header.respChannel);
    resp->header.op = kv_put;
    resp->header.relId = key->tableOid;

    free(key);
    free(value);

    return resp;
}
