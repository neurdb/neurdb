#ifndef NRAM_H
#define NRAM_H

#include "kv_access/kv.h"
#include "kv_storage/rocksengine.h"
#include "test/kv_test.h"

typedef struct NRAMState {
    KVEngine *engine;       // Pointer to the RocksDB backend
    int *key_attrs;         // Array of key attribute numbers
    int nkeys;              // Number of primary key attributes
} NRAMState;


#endif //NRAM_H
