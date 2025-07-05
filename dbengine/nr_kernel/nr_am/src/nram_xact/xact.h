/* -------------------------------------------------------------------------
 * xact.h
 * NRAM transaction management functions.
 *
 * -------------------------------------------------------------------------
 */
#ifndef NRAMXACT_H
#define NRAMXACT_H

#include "postgres.h"
#include "nram_access/kv.h"
#include "access/xact.h"
#include "nodes/pg_list.h"

typedef enum XactOpType {
    XACT_OP_READ,
    XACT_OP_WRITE,
    XACT_OP_DELETE,
    XACT_OP_INSERT,
    XACT_OP_SCAN
} XactOpType;

typedef struct NRAMXactOptData {
    XactOpType type;
    TransactionId xact_id;
    NRAMKey key;
    NRAMValue value;
} NRAMXactOptData;

typedef NRAMXactOptData *NRAMXactOpt;

typedef struct NRAMXactStateData {
    TransactionId xact_id;
    bool validated;
    TimestampTz begin_ts;
    // rocksdb_snapshot_t *snapshot;

    List *read_set;     // key hash -> version or just existence
    List *write_set;
    // All updates are bufferred, and only flushed during transaction commit.
} NRAMXactStateData;

typedef NRAMXactStateData *NRAMXactState;


extern void refresh_nram_xact(void);
extern void nram_register_xact_hook(void);
extern void nram_unregister_xact_hook(void);
extern NRAMXactState NewNRAMXactState(TransactionId xact_id);
extern NRAMXactOpt find_read_set(NRAMXactState state, NRAMKey key);
extern NRAMXactOpt find_write_set(NRAMXactState state, NRAMKey key);
extern void add_read_set(NRAMXactState state, NRAMKey key, NRAMValue value);
extern void add_write_set(NRAMXactState state, NRAMKey key, NRAMValue value);
extern bool read_own_write(NRAMXactState state, const NRAMKey key, NRAMValue *value);
extern bool read_own_read(NRAMXactState state, const NRAMKey key, NRAMValue *value);
extern bool validate_read_set(NRAMXactState state);
extern NRAMXactState GetCurrentNRAMXact(void);

#endif
