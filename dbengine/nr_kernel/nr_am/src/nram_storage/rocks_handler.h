#include "nram_access/kv.h"

NRAMValue RocksClientGet(uint32 tableOid, NRAMKey key);
bool RocksClientPut(uint32 tableOid, NRAMKey key, NRAMValue value);
void CloseRespChannel(void);

