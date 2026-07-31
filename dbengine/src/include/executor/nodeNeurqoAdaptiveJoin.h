/*-------------------------------------------------------------------------
 *
 * nodeNeurqoAdaptiveJoin.h
 *	  Executor support for NeurQO runtime adaptive joins.
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODENEURQOADAPTIVEJOIN_H
#define NODENEURQOADAPTIVEJOIN_H

#include "nodes/plannodes.h"

typedef struct NeurqoAdaptiveJoinStats
{
	int			joins_decided;
	int			nestloop_selected;
	int			hashjoin_selected;
	uint64		actual_build_rows;
	double		build_ms;
} NeurqoAdaptiveJoinStats;

extern int neurqo_wrap_adaptive_joins(PlannedStmt *baseline,
									  PlannedStmt *nestloop_alternative,
									  const char *level,
									  int threshold_rows,
									  int max_nestloop_cost_ratio_pct,
									  uint64 run_id,
									  int round);
extern int neurqo_count_adaptive_hashjoins(PlannedStmt *plannedstmt);
extern void neurqo_reset_adaptive_join_stats(void);
extern NeurqoAdaptiveJoinStats neurqo_get_adaptive_join_stats(void);

#endif							/* NODENEURQOADAPTIVEJOIN_H */
