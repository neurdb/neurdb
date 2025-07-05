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
    if (RespChannel) {
        KVChannelDestroy(RespChannel);
        RespChannel = NULL;
    }
}

bool RocksClientPut(NRAMKey key, NRAMValue value) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len, val_len, total_len;
    char *serialized_key = tkey_serialize(key, &key_len);
    char *serialized_val = tvalue_serialize(value, &val_len);
    KVMsg *msg = NewMsg(kv_put, key->tableOid, kv_status_none, MyProcPid), *resp;
    bool ok, success;

    total_len = key_len + val_len + sizeof(Size);

    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    memcpy(msg->entity, &key_len, sizeof(Size));
    memcpy((char *)msg->entity + sizeof(Size), serialized_key, key_len);
    memcpy((char *)msg->entity + sizeof(Size) + key_len, serialized_val, val_len);

    ok = KVChannelPushMsg(req_chan, msg, true);
    Assert(ok);

    resp = KVChannelPopMsg(resp_chan, true);
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

    msg->header.entitySize = key_len;
    msg->entity = serialized_key;

    ok = KVChannelPushMsg(req_chan, msg, true);
    Assert(ok);

    resp = KVChannelPopMsg(resp_chan, true);

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
