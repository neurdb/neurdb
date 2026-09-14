#ifndef NRINDEX_H
#define NRINDEX_H

#include "postgres.h"
#include "access/amapi.h"
#include "access/relscan.h"
#include "nodes/pathnodes.h"
#include "storage/block.h"
#include "utils/rel.h"

/* Include NRAM dependencies */
#include "nram_access/kv.h"
#include "nram_storage/rocks_service.h"
#include "nram_storage/rocks_handler.h"

/* Index scan descriptor for nrindex */
typedef struct NRIndexScanDescData *NRIndexScanDesc;

/* Function declarations */
extern Datum nrindex_handler(PG_FUNCTION_ARGS);

/* Module lifecycle functions */
extern void _PG_init(void);
extern void _PG_fini(void);

#endif /* NRINDEX_H */
