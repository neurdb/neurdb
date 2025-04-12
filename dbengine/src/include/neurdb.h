#ifndef NEURDB_H
#define NEURDB_H

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
