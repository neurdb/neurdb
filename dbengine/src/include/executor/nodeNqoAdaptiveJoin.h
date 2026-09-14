/*-------------------------------------------------------------------------
 *
 * nodeNqoAdaptiveJoin.h
 *	  Executor support for NQO runtime adaptive joins.
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODENQOADAPTIVEJOIN_H
#define NODENQOADAPTIVEJOIN_H

#include "nodes/plannodes.h"

typedef struct NqoAdaptiveJoinStats
{
	int			joins_decided;
	int			nestloop_selected;
	int			hashjoin_selected;
	uint64		actual_build_rows;
	double		build_ms;
} NqoAdaptiveJoinStats;

extern int nqo_wrap_adaptive_joins(PlannedStmt *baseline,
									  PlannedStmt *nestloop_alternative,
									  const char *level,
									  int threshold_rows,
									  int max_nestloop_cost_ratio_pct,
									  uint64 run_id,
									  int round);
extern int nqo_count_adaptive_hashjoins(PlannedStmt *plannedstmt);
extern void nqo_reset_adaptive_join_stats(void);
extern NqoAdaptiveJoinStats nqo_get_adaptive_join_stats(void);

#endif							/* NODENQOADAPTIVEJOIN_H */
