#define ROCKSDB_PATH "pg_rocksdb"
#define ROCKSDB_CHANNEL "rocks_service_channel"
#define ROCKSDB_CHANNEL_TIMEOUT 1000 // (ms)

// PHX: use the following two debug macros when debugging the code.
// #define NRAM_TEST_INFO(fmt, ...) elog(INFO, "[NRAM] [%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
// #define NRAM_INFO() elog(INFO, "[NRAM] calling function %s", __func__)


#define NRAM_TEST_INFO(fmt, ...)
#define NRAM_INFO()
