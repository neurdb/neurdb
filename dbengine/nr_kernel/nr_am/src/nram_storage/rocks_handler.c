#include "rocks_handler.h"
#include "nram_storage/rocks_service.h"

#define RESP_CHAN() Assert(RespChannel != NULL), RespChannel

static KVChannel *RespChannel = NULL;
static KVChannel *ServerChannel = NULL;


static inline KVChannel* GetRespChannel(void) {
    char resp_name[NAMEDATALEN];

    if (RespChannel != NULL)
        return RespChannel;

    snprintf(resp_name, sizeof(resp_name), "kv_resp_%d", MyProcPid);
    RespChannel = KVChannelInit(resp_name, true);
    return RespChannel;
}

static inline KVChannel* GetServerChannel(void) {
    if (ServerChannel != NULL)
        return ServerChannel;

    ServerChannel = KVChannelInit(ROCKSDB_CHANNEL, false);
    return ServerChannel;
}

void CloseRespChannel(void) {
    KVChannelDestroy(RespChannel);
    RespChannel = NULL;
}

bool RocksClientPut(NRAMKey key, NRAMValue value) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len, val_len, total_len;
    char *serialized_key = tkey_serialize(key, &key_len);
    char *serialized_val = tvalue_serialize(value, &val_len);
    KVMsg *msg = NewMsg(kv_put, key->tableOid, kv_status_none, MyProcPid), *resp;
    bool ok, success;

    NRAM_INFO();

    total_len = key_len + val_len + sizeof(Size);

    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    memcpy(msg->entity, &key_len, sizeof(Size));
    memcpy((char *)msg->entity + sizeof(Size), serialized_key, key_len);
    memcpy((char *)msg->entity + sizeof(Size) + key_len, serialized_val, val_len);

    ok = KVChannelPushMsg(req_chan, msg, -1);
    if (!ok) {
        elog(WARNING, "RocksClientPut: message pushing failed.");
        return false;
    }

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_put;

    if (!success) {
        PrintKVMsg(resp);
        elog(WARNING, "[NRAM] Rocks PUT failed");
    }

    pfree(serialized_key);
    pfree(serialized_val);
    pfree(msg->entity);
    pfree(msg);
    if (resp) {
        if (resp->entity) pfree(resp->entity);
        pfree(resp);
    }
    return success;
}


NRAMValue RocksClientGet(NRAMKey key) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len;
    char *serialized_key = tkey_serialize(key, &key_len);
    KVMsg *msg = NewMsg(kv_get, key->tableOid, kv_status_none, MyProcPid), *resp;
    bool ok, success;
    NRAMValue val_out;

    NRAM_INFO();

    msg->header.entitySize = key_len;
    msg->entity = serialized_key;

    ok = KVChannelPushMsg(req_chan, msg, -1);
    if (!ok) {
        elog(WARNING, "RocksClientGet: message pushing failed.");
        return NULL;
    }

    resp = KVChannelPopMsg(resp_chan, -1);

    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_get;
    if (!success) {
        PrintKVMsg(resp);
        elog(ERROR, "[NRAM] Rocks GET failed");
    }

    val_out = tvalue_deserialize((char *)resp->entity, resp->header.entitySize);

    pfree(serialized_key);
    pfree(msg);
    if (resp) {
        if (resp->entity) pfree(resp->entity);
        pfree(resp);
    }

    return val_out;
}


// Note: the range is fetched from a snapshot!
bool RocksClientRangeScan(NRAMKey start_key, NRAMKey end_key,
                          NRAMKey **out_keys, NRAMValue **out_values, uint32_t *out_count) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len_1, key_len_2, total_len;
    char *ptr;
    KVMsg *msg, *resp;
    bool ok, success;

    char *serialized_start = tkey_serialize(start_key, &key_len_1);
    char *serialized_end = tkey_serialize(end_key, &key_len_2);

    total_len = sizeof(Size) + key_len_1 + sizeof(Size) + key_len_2;

    msg = NewMsg(kv_range, start_key->tableOid, kv_status_none, MyProcPid);
    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    ptr = msg->entity;
    memcpy(ptr, &key_len_1, sizeof(Size));
    ptr += sizeof(Size);
    memcpy(ptr, serialized_start, key_len_1);
    ptr += key_len_1;
    memcpy(ptr, &key_len_2, sizeof(Size));
    ptr += sizeof(Size);
    memcpy(ptr, serialized_end, key_len_2);

    ok = KVChannelPushMsg(req_chan, msg, -1);
    if (!ok) {
        elog(WARNING, "RocksClientRangeScan: message pushing failed.");
        return false;
    }

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_range;

    if (success) {
        ptr = resp->entity;
        memcpy(out_count, ptr, sizeof(int));
        ptr += sizeof(int);

        *out_keys = palloc(sizeof(NRAMKey) * (*out_count));
        *out_values = palloc(sizeof(NRAMValue) * (*out_count));

        for (int i = 0; i < *out_count; i++) {
            Size klen, vlen;
            memcpy(&klen, ptr, sizeof(Size));
            ptr += sizeof(Size);
            (*out_keys)[i] = tkey_deserialize(ptr, klen);
            ptr += klen;

            memcpy(&vlen, ptr, sizeof(Size));
            ptr += sizeof(Size);
            (*out_values)[i] = tvalue_deserialize(ptr, vlen);
            ptr += vlen;
        }
    } else {
        elog(WARNING, "[NRAM] Rocks RANGE_SCAN with keys failed");
    }

    pfree(serialized_start);
    pfree(serialized_end);
    pfree(msg->entity);
    pfree(msg);
    if (resp) {
        if (resp->entity) pfree(resp->entity);
        pfree(resp);
    }
    return success;
}
