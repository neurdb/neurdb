#ifndef NRAMPOLICY_H
#define NRAMPOLICY_H

#include "postgres.h"
#include "storage/lockdefs.h"
#include "storage/backendid.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "lib/stringinfo.h"
#include "storage/proc.h"


/* Helper functions */

#define NUM_OF_SYS_XACTS 5
#define SKIP_XACT(tid) ((tid) <= NUM_OF_SYS_XACTS)
#define SEC_TO_NS(sec) ((sec)*1000000000)
#define NS_TO_US(ns) ((ns)/1000.0)
#define DEFAULT_WAIT_NS SEC_TO_NS(1)  // 1 second.

#define READ_OPT 0
#define UPDATE_OPT 1
#define MAX_XACT_FEATURE_SPACE (1<<10)


/* ------------------------------------------------------------------------
 * NeurCC related codes.
 * ------------------------------------------------------------------------
 */

typedef struct CCActionData {
   bool     detect_all;
   double   priority;
   uint32_t timeout; // in nanoseconds.
} CCActionData;

typedef CCActionData* CCAction;


typedef struct CachedAgentFuncData {
   CCActionData act[MAX_XACT_FEATURE_SPACE];
} CachedAgentFuncData;

typedef CachedAgentFuncData* CachedAgentFunc;

extern CachedAgentFunc GlobalCachedAgentFunc;


typedef struct XactFeatureData {
   // transactional information.
   uint32_t n_access; // the number of operations.
   uint8 cur_op;   // current operation type, read or write.
   // graphical information.
   uint16_t n_dep; // dependency graph information.
} XactFeatureData;

typedef XactFeatureData* XactFeature;

extern void init_agent_function_cache();
extern void load_agent_function(const char *filename);
extern CCAction get_action(XactFeature feature);
extern void print_xact_feature(StringInfo str, XactFeature feature);

#endif //NRAMPOLICY_H