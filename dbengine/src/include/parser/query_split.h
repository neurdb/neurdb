/*-------------------------------------------------------------------------
 *
 *
 *
 *
 * IDENTIFICATION
 *	  src/include/parser/query_split.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef QUERY_SPLIT_H
#define QUERY_SPLIT_H

#include "postgres.h"

#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/time.h>
#include <sys/resource.h>
#endif
#include "access/heapam.h"
#include "access/parallel.h"
#include "access/printtup.h"
#include "access/table.h"
#include "access/xact.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "commands/async.h"
#include "commands/createas.h"
#include "commands/matview.h"
#include "commands/prepare.h"
#include "executor/spi.h"
#include "jit/jit.h"
#include "libpq/libpq.h"
#include "libpq/pqformat.h"
#include "libpq/pqsignal.h"
#include "miscadmin.h"
#include "nodes/parsenodes.h"
#include "nodes/pathnodes.h"
#include "nodes/pg_list.h"
#include "nodes/print.h"
#include "nodes/makefuncs.h"
#include "optimizer/optimizer.h"
#include "pgstat.h"
#include "pg_trace.h"
#include "parser/analyze.h"
#include "parser/parser.h"
#include "pg_getopt.h"
#include "postmaster/autovacuum.h"
#include "postmaster/postmaster.h"
#include "replication/logicallauncher.h"
#include "replication/logicalworker.h"
#include "replication/slot.h"
#include "replication/walsender.h"
#include "rewrite/rewriteHandler.h"
#include "storage/bufmgr.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/procsignal.h"
#include "storage/sinval.h"
#include "tcop/fastpath.h"
#include "tcop/pquery.h"
#include "tcop/tcopprot.h"
#include "tcop/utility.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/ps_status.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/timeout.h"
#include "utils/timestamp.h"
#include "mb/pg_wchar.h"

/* query-splitting algorithm modes (ported from original querysplit postgres.h) */
#define None 0
#define Minsubquery 1
#define RelationshipCenter 2
#define EntityCenter 3
/* execution-order decision (the "alpha" knob) */
#define only_cost 0
#define only_row 1
#define hybrid_row 2
#define hybrid_sqrt 3
#define hybrid_log 4
#define global_view 5

/* set by the GUC/AI policy; read inside query_split.c */
extern int query_splitting_algorithm;
extern int order_decision;
extern bool neurqo_enabled;		/* the `neurqo` on/off GUC */
extern char *neurqo_server_url;
extern char *neurqo_trajectory_log_path;
extern int neurqo_server_timeout_ms;
extern int neurqo_max_rounds;
extern int neurqo_search_topk;
extern int neurqo_search_max_rels;

/* DR_intorel is static in PG16's createas.c; expose it here for Prepare4Next.
 * Layout copied verbatim from src/backend/commands/createas.c (PG16). */
typedef struct
{
	DestReceiver pub;
	IntoClause *into;
	Relation	rel;
	ObjectAddress reladdr;
	CommandId	output_cid;
	int			ti_options;
	BulkInsertState bistate;
} DR_intorel;

void doQSparse(const char* query_string, CommandTag commandTag, Node* pstmt, Query* querytree, QueryCompletion* completionTag);

#endif
