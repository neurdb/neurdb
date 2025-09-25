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
#include "nodes/parsenodes.h"
#include "nram_xact/action.h"
#include "executor/executor.h"

#define nram_for_modify(x) (x && (x->cur_cmdtype == CMD_UPDATE || x->cur_cmdtype == CMD_DELETE || x->cur_cmdtype == CMD_INSERT))
#define nram_for_read(x) (x && (x->cur_cmdtype == CMD_SELECT))

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

#define NRAM_CMD_STACK_MAX 8

typedef struct NRAMXactStateData {
    TransactionId xact_id;
    bool validated;
    TimestampTz begin_ts;
    // rocksdb_snapshot_t *snapshot;
    XactFeature feature;    // used for NeurCC
    CCAction action;      // used for NeurCC
    CmdType cur_cmdtype;                 /* CMD_SELECT / CMD_UPDATE / ... */
    int     cmdtype_depth;
    CmdType cmdtype_stack[NRAM_CMD_STACK_MAX];

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

extern void before_access(NRAMXactState state);
extern void nram_lock_for_read(Relation relation, ItemPointer tid);
extern void nram_lock_for_write(Relation relation, ItemPointer tid);
extern void nram_lock_for_scan(Relation relation);
extern LockAcquireResult nram_try_lock(NRAMKey key, LOCKMODE mode);

#endif
