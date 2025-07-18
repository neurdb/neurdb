#ifndef NR_GUC_H
#define NR_GUC_H

#include "postgres.h"

/*
 * GUC variable for current configuration
 */
extern PGDLLIMPORT char *NrModelName;
extern PGDLLIMPORT int NrTaskBatchSize;
extern PGDLLIMPORT int NrTaskEpoch;
extern PGDLLIMPORT int NrTaskMaxFeatures;
extern PGDLLIMPORT int NrTaskNumBatches;

#endif
