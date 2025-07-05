#include "nram_access/kv.h"

NRAMValue RocksClientGet(NRAMKey key);
bool RocksClientPut(NRAMKey key, NRAMValue value);
void CloseRespChannel(void);
bool RocksClientRangeScan(NRAMKey start_key, NRAMKey end_key, NRAMKey **out_keys, NRAMValue **out_results, int *out_count);

