#include "nram_access/kv.h"
#include "nrindex_access/nrindex_kv.h"

NRAMValue RocksClientGet(NRAMKey key);
bool RocksClientPut(NRAMKey key, NRAMValue value);
void CloseRespChannel(void);
bool RocksClientRangeScan(NRAMKey start_key, NRAMKey end_key, NRAMKey **out_keys, NRAMValue **out_results, uint32_t *out_count);

/* Index operations */
NRIndexValue RocksClientIndexGet(NRIndexKey ikey);
bool RocksClientIndexPut(NRIndexKey ikey, NRIndexValue ivalue);
bool RocksClientIndexDelete(NRIndexKey ikey);
bool RocksClientIndexRangeScan(NRIndexKey start_key, NRIndexKey end_key, NRIndexKey **out_keys, NRIndexValue **out_results, int *out_count);

/* Bulk load for efficient index building (supports INT and BIGINT) */
bool RocksClientIndexBulkLoad(Oid indexOid, int64 *keys, uint64 *values, int count);
