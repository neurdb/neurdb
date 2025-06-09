#include "postgres.h"

#include "neurdb/guc.h"

/**
 * Configurable parameters
 *
 * Set in `backend/utils/misc/guc_tables.c`
 */
char *NrModelName = NULL;
int NrTaskBatchSize = 128;
int NrTaskEpoch = 1;
int NrTaskMaxFeatures = 10;
int NrTaskNumBatches = 30;
