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


typedef struct NRAMXactOptData {
    // CmdType type;
    NRAMKey *key;
    NRAMValue *value;
} NRAMXactOptData;

typedef NRAMXactOptData *NRAMXactOpt;

typedef struct NRAMXactStateData {
    TransactionId tid;
    bool validated;
    TimestampTz begin_ts;
    // rocksdb_snapshot_t *snapshot;

    NRAMXactOpt *read_set;     // key hash -> version or just existence
    NRAMXactOpt *write_set;    // key hash -> new value or pending write
} NRAMXactStateData;

typedef NRAMXactStateData *NRAMXactState;


extern void refresh_nram_xact(void);
extern void nram_register_xact_hook(void);
extern void nram_unregister_xact_hook(void);
extern NRAMXactState NewNRAMXactState(TransactionId tid);

#endif