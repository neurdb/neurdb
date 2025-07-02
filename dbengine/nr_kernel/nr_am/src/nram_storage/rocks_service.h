#ifndef ROCKS_SERVICE_H
#define ROCKS_SERVICE_H

#include "nram_storage/rocksengine.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"

#define ROCKS_RUNNING_BIT 1

void run_rocks(int num_threads);
void run_rocks_no_thread(void);

void *process_request(void *arg);

KVMsg *handle_kv_get(KVMsg *msg);
KVMsg *handle_kv_put(KVMsg *msg);
  
void nram_rocks_service_init(void);
void nram_rocks_service_terminate(void);

#endif // ROCKS_SERVICE_H
