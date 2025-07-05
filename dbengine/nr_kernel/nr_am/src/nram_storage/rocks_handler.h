#include "nram_access/kv.h"

NRAMValue RocksClientGet(NRAMKey key);
bool RocksClientPut(NRAMKey key, NRAMValue value);
void CloseRespChannel(void);

