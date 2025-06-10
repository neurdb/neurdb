#ifndef NRAM_H
#define NRAM_H

#include "nram_access/kv.h"
#include "nram_storage/rocksengine.h"
#include "nram_xact/xact.h"
#include "test/kv_test.h"


#define NRAM_STATE_MAGIC 0x4E52414D  // 'NRAM'
#define ASSERT_VALID_NRAM_STATE(ptr)                                            \
        if ((ptr) == NULL || (ptr)->magic != NRAM_STATE_MAGIC)                 \
            elog(ERROR, "[NRAM] Invalid or corrupted NRAMState pointer: %p", (void *)(ptr));
#define IS_VALID_NRAM_STATE(ptr) ((ptr) != NULL && (ptr)->magic == NRAM_STATE_MAGIC)
#define NRAM_XACT_BEGIN_BLOCK refresh_nram_xact()

typedef struct NRAMState {
    uint32 magic;           // DEBUG bits for nram state, check memory corruptions.
    KVEngine *engine;       // Pointer to the RocksDB backend
} NRAMState;

void nram_shutdown_session(void);
List *nram_get_primary_key_attrs(Relation rel);


#endif //NRAM_H
