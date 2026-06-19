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

/* PREDICT operator cost-model parameters (see cost_neurdbpredict) */
extern PGDLLIMPORT double NrPredictStartupCost;
extern PGDLLIMPORT double NrPredictTupleCost;
extern PGDLLIMPORT double NrPredictBatchCost;

/* allow pushing input-column quals below the PREDICT operator */
extern PGDLLIMPORT bool NrPredictPushdown;

/*
 * allow pulling PREDICT above outer inner-join filters by merging those joins
 * into the PREDICT input relation
 */
extern PGDLLIMPORT bool NrPredictPullup;

#endif
