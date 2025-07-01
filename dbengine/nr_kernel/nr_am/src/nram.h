#ifndef NRAM_H
#define NRAM_H

#include "nram_access/kv.h"
#include "nram_storage/rocksengine.h"
#include "nram_xact/xact.h"
#include "test/kv_test.h"
#include "test/channel_test.h"

#define NRAM_XACT_BEGIN_BLOCK refresh_nram_xact()

void nram_shutdown_session(void);

#endif //NRAM_H
