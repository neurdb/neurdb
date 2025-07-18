#include "postgres.h"

#include "neurdb/guc.h"

/**
 * Configurable parameters
 *
 * Set in `backend/utils/misc/guc_tables.c`
 */
char *NrModelName = NULL;
int NrTaskBatchSize;
int NrTaskEpoch;
int NrTaskMaxFeatures;
int NrTaskNumBatches;
