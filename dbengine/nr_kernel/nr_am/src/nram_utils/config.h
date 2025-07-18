#include "storage/pmsignal.h"

#define ROCKSDB_PATH "pg_rocksdb"
#define ROCKSDB_CHANNEL "rocks_service_channel"
#define ROCKSDB_CHANNEL_NO_TIMEOUT 0
#define ROCKSDB_CHANNEL_DEFAULT_TIMEOUT 1 // (ms)
#define MAX_PROC_COUNT 16

#define NRAM_UNSUPPORTED() elog(WARNING, "[NRAM] calling unsupported function %s", __func__)

// PHX: use the following two debug macros when debugging the code.
// #define NRAM_TEST_INFO(fmt, ...) elog(INFO, "[NRAM] [%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
// #define NRAM_INFO() elog(INFO, "[NRAM] calling function %s", __func__)


#define NRAM_TEST_INFO(fmt, ...)
#define NRAM_INFO()
