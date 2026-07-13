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

/*
 * Planner cost parameters for the PREDICT (NeurDBPredict) operator, in the
 * same arbitrary units as seq_page_cost (1.0) / cpu_tuple_cost (0.01).
 * Consumed by cost_neurdbpredict() in optimizer/path/costsize.c.
 */
double NrPredictStartupCost;	/* engine session + model/context setup */
double NrPredictTupleCost;		/* per input row inferred */
double NrPredictBatchCost;		/* per batch round trip to the AI engine */

/*
 * Allow the planner to push input-column quals below the PREDICT operator
 * (cost-based dynamic scheduling).  Off = the operator always stays at the
 * root of its subquery and all quals are evaluated above it.
 */
bool NrPredictPushdown;
