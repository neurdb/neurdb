#ifndef NRAM_H
#define NRAM_H

#include "kv_access/kv.h"
#include "kv_storage/rocksengine.h"
#include "test/kv_test.h"

#define NRAM_STATE_MAGIC 0x4E52414D  // 'NRAM'
#define ASSERT_VALID_NRAM_STATE(ptr)                                            \
        if ((ptr) == NULL || (ptr)->magic != NRAM_STATE_MAGIC)                 \
            elog(ERROR, "[NRAM] Invalid or corrupted NRAMState pointer: %p", (void *)(ptr));
#define IS_VALID_NRAM_STATE(ptr) ((ptr) != NULL && (ptr)->magic == NRAM_STATE_MAGIC)

typedef struct NRAMState {
    uint32 magic;           // DEBUG bits for nram state, check memory corruptions.
    KVEngine *engine;       // Pointer to the RocksDB backend
    int nkeys;              // Number of primary key attributes
    int *key_attrs;         // Array of key attribute numbers
} NRAMState;


#endif //NRAM_H
