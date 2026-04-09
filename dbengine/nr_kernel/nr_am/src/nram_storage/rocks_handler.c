#include "rocks_handler.h"
#include "nram_storage/rocks_service.h"

#define RESP_CHAN() Assert(RespChannel != NULL), RespChannel

static KVChannel *RespChannel = NULL;
static KVChannel *ServerChannel = NULL;


static inline KVChannel* GetRespChannel(void) {
    char resp_name[NAMEDATALEN];
    NRAM_INFO();

    if (RespChannel != NULL)
        return RespChannel;

    snprintf(resp_name, sizeof(resp_name), "kv_resp_%d", MyProcPid);
    RespChannel = KVChannelInit(resp_name, true);
    return RespChannel;
}

static inline KVChannel* GetServerChannel(void) {
    NRAM_INFO();
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
        memcpy(out_count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

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

/* ------------------------------------------------------------------------
 * Index operations
 * ------------------------------------------------------------------------
 */

NRIndexValue RocksClientIndexGet(NRIndexKey ikey) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len;
    char *serialized_key = nrindex_key_serialize(ikey, &key_len);
    KVMsg *msg = NewMsg(kv_index_get, ikey->indexOid, kv_status_none, MyProcPid), *resp;
    NRIndexValue result = NULL;

    NRAM_INFO();

    msg->header.entitySize = key_len;
    msg->entity = serialized_key;

    if (!KVChannelPushMsg(req_chan, msg, -1)) {
        elog(WARNING, "RocksClientIndexGet: message pushing failed.");
        return NULL;
    }

    resp = KVChannelPopMsg(resp_chan, -1);
    if (resp && resp->header.status == kv_status_ok && resp->header.op == kv_index_get) {
        if (resp->entity && resp->header.entitySize > 0) {
            result = nrindex_value_deserialize(resp->entity, resp->header.entitySize);
        }
    } else {
        elog(WARNING, "[NRAM] Rocks INDEX_GET failed");
    }

    pfree(serialized_key);
    pfree(msg);
    if (resp) {
        if (resp->entity) pfree(resp->entity);
        pfree(resp);
    }
    return result;
}

bool RocksClientIndexPut(NRIndexKey ikey, NRIndexValue ivalue) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len, val_len, total_len;
    // 步骤1: 序列化 Key 和 Value
    char *serialized_key = nrindex_key_serialize(ikey, &key_len);
    char *serialized_val = nrindex_value_serialize(ivalue, &val_len);
    KVMsg *msg = NewMsg(kv_index_put, ikey->indexOid, kv_status_none, MyProcPid), *resp;
    bool success;

    NRAM_INFO();

    total_len = key_len + val_len + 2 * sizeof(Size);

    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    // 步骤2: 构建 IPC 消息
    char *ptr = msg->entity;
    memcpy(ptr, &key_len, sizeof(Size));
    ptr += sizeof(Size);
    memcpy(ptr, serialized_key, key_len);
    ptr += key_len;
    memcpy(ptr, &val_len, sizeof(Size));
    ptr += sizeof(Size);
    memcpy(ptr, serialized_val, val_len);

    // 步骤3: 发送到共享内存通道
    if (!KVChannelPushMsg(req_chan, msg, -1)) {
        elog(WARNING, "RocksClientIndexPut: message pushing failed.");
        return false;
    }

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_index_put;

    if (!success) {
        elog(WARNING, "[NRAM] Rocks INDEX_PUT failed");
    }

    pfree(serialized_key);
    pfree(serialized_val);
    pfree(msg->entity);
    pfree(msg);
    if (resp) {
        pfree(resp);
    }
    return success;
}

bool RocksClientIndexDelete(NRIndexKey ikey) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size key_len;
    char *serialized_key = nrindex_key_serialize(ikey, &key_len);
    KVMsg *msg = NewMsg(kv_index_delete, ikey->indexOid, kv_status_none, MyProcPid), *resp;
    bool success;

    NRAM_INFO();

    msg->header.entitySize = key_len;
    msg->entity = serialized_key;

    if (!KVChannelPushMsg(req_chan, msg, -1)) {
        elog(WARNING, "RocksClientIndexDelete: message pushing failed.");
        return false;
    }

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_index_delete;

    if (!success) {
        elog(WARNING, "[NRAM] Rocks INDEX_DELETE failed");
    }

    pfree(serialized_key);
    pfree(msg);
    if (resp) {
        pfree(resp);
    }
    return success;
}

bool RocksClientIndexRangeScan(NRIndexKey start_key, NRIndexKey end_key,
                               NRIndexKey **out_keys, NRIndexValue **out_results, int *out_count) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size start_len = 0, end_len = 0, total_len;
    char *serialized_start = NULL, *serialized_end = NULL;
    Oid indexOid;
    KVMsg *msg, *resp;
    bool success;
    char *ptr;

    NRAM_INFO();

    /* Handle NULL keys - use length 0 to indicate NULL */
    if (start_key) {
        serialized_start = nrindex_key_serialize(start_key, &start_len);
        indexOid = start_key->indexOid;
    } else if (end_key) {
        indexOid = end_key->indexOid;
    } else {
        elog(WARNING, "RocksClientIndexRangeScan: both keys are NULL");
        return false;
    }

    if (end_key) {
        serialized_end = nrindex_key_serialize(end_key, &end_len);
    }

    msg = NewMsg(kv_index_range_scan, indexOid, kv_status_none, MyProcPid);

    total_len = 2 * sizeof(Size) + start_len + end_len;

    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    ptr = msg->entity;
    memcpy(ptr, &start_len, sizeof(Size));
    ptr += sizeof(Size);
    if (start_len > 0) {
        memcpy(ptr, serialized_start, start_len);
        ptr += start_len;
    }
    memcpy(ptr, &end_len, sizeof(Size));
    ptr += sizeof(Size);
    if (end_len > 0) {
        memcpy(ptr, serialized_end, end_len);
    }

    if (serialized_start) pfree(serialized_start);
    if (serialized_end) pfree(serialized_end);

    elog(NOTICE, "[Client] RocksClientIndexRangeScan: sending message, start_len=%zu, end_len=%zu, total_len=%zu",
         start_len, end_len, total_len);

    if (!KVChannelPushMsg(req_chan, msg, -1)) {
        elog(WARNING, "RocksClientIndexRangeScan: message pushing failed.");
        return false;
    }

    elog(NOTICE, "[Client] RocksClientIndexRangeScan: message sent, waiting for response...");

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_index_range_scan;

    if (success) {
        ptr = resp->entity;
        memcpy(out_count, ptr, sizeof(int));
        ptr += sizeof(int);

        *out_keys = palloc(sizeof(NRIndexKey) * (*out_count));
        *out_results = palloc(sizeof(NRIndexValue) * (*out_count));

        for (int i = 0; i < *out_count; i++) {
            Size klen, vlen;
            memcpy(&klen, ptr, sizeof(Size));
            ptr += sizeof(Size);
            (*out_keys)[i] = nrindex_key_deserialize(ptr, klen);
            ptr += klen;

            memcpy(&vlen, ptr, sizeof(Size));
            ptr += sizeof(Size);
            (*out_results)[i] = nrindex_value_deserialize(ptr, vlen);
            ptr += vlen;
        }
    } else {
        elog(WARNING, "[NRAM] Rocks INDEX_RANGE_SCAN failed");
    }

    /* Note: serialized_start and serialized_end were already freed at lines 357-358 */
    pfree(msg->entity);
    pfree(msg);
    if (resp) {
        if (resp->entity) pfree(resp->entity);
        pfree(resp);
    }
    return success;
}

bool RocksClientIndexBulkLoad(Oid indexOid, int64 *keys, uint64 *values, int count) {
    KVChannel *req_chan = GetServerChannel(), *resp_chan = GetRespChannel();
    Size total_len;
    KVMsg *msg, *resp;
    bool success;
    char *ptr;

    NRAM_INFO();

    elog(LOG, "[Client] RocksClientIndexBulkLoad: indexOid=%u, count=%d", indexOid, count);

    /* Message format: [count (4 bytes)] [keys (count * 8 bytes)] [values (count * 8 bytes)] */
    total_len = sizeof(int) + (count * sizeof(int64)) + (count * sizeof(uint64));

    msg = NewMsg(kv_index_bulk_load, indexOid, kv_status_none, MyProcPid);
    msg->header.entitySize = total_len;
    msg->entity = palloc(total_len);

    ptr = msg->entity;

    /* Write count */
    memcpy(ptr, &count, sizeof(int));
    ptr += sizeof(int);

    /* Write keys array */
    memcpy(ptr, keys, count * sizeof(int64));
    ptr += count * sizeof(int64);

    /* Write values array */
    memcpy(ptr, values, count * sizeof(uint64));

    if (!KVChannelPushMsg(req_chan, msg, -1)) {
        elog(WARNING, "RocksClientIndexBulkLoad: message pushing failed.");
        pfree(msg->entity);
        pfree(msg);
        return false;
    }

    elog(LOG, "[Client] RocksClientIndexBulkLoad: message sent, waiting for response...");

    resp = KVChannelPopMsg(resp_chan, -1);
    success = resp && resp->header.status == kv_status_ok && resp->header.op == kv_index_bulk_load;

    if (!success) {
        elog(WARNING, "[NRAM] Rocks INDEX_BULK_LOAD failed");
    } else {
        elog(LOG, "[Client] RocksClientIndexBulkLoad: success");
    }

    pfree(msg->entity);
    pfree(msg);
    if (resp) {
        pfree(resp);
    }
    return success;
}
