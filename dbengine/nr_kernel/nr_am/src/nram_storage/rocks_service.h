#ifndef ROCKS_SERVICE_H
#define ROCKS_SERVICE_H

#include "nram_storage/rocksengine.h"
#include "ipc/msg.h"

#define ROCKS_RUNNING_BIT 1

void run_rocks(int num_threads);

void *process_request(void *arg);

KVMsg *handle_kv_get(KVMsg *msg);
KVMsg *handle_kv_put(KVMsg *msg);

#endif // ROCKS_SERVICE_H
