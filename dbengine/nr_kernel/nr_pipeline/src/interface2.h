/*-------------------------------------------------------------------------
 *
 * interface2.c
 *	  new interface (tuple-level) implementation for nr_pipeline
 *
 * ORIGINAL AUTHOR: Siqi Xiang
 *
 *-------------------------------------------------------------------------
 */
#ifndef INTERFACE2_H
#define INTERFACE2_H

#include <postgres.h>
#include <fmgr.h>

#include <executor/spi.h>

#include "utils/network/websocket.h"

struct DistributedInfer;

typedef enum {
    PS_UNINIT = 0,
    PS_TRAIN = 1,
    PS_INFER = 2
} PipelineState;

typedef struct {
    PipelineState state;
    NrWebsocket *ws;
    int model_id;
    // Task spec
    char *model_name;
    char *table_name;
    int batch_size; // call inferenc every batch_size
    int n_batches;
    int epoch;
    int nfeat;
    PredictType type;

    int n_features;
    char **feature_names;
    char *target;
    TupleDesc tupdesc;
    int label_col_id;

    // number of batches for train, eval, test
    int nb_tr, nb_ev, nb_te;

    // batch data
    HeapTuple *batch_vals;
    int batch_count;
    int batch_capacity;

    MemoryContext hash_ctx;
    HTAB *class_id_map;
    List *id_class_map;
    struct DistributedInfer *dist_infer;

    /*
     * Broadcast-train state for in-context models (tabpfn): the fitted
     * context lives inside one AI-server process keyed by a process-local
     * model id, so to shard inference across engines the SAME context is
     * fitted on EVERY registered engine during the train phase, and the
     * per-engine model ids are remembered for the inference task.
     */
    int eng_count;            /* engines captured at train time (0 = none) */
    char **eng_hosts;
    int *eng_ports;
    NrWebsocket **train_wss;  /* one ws per engine while PS_TRAIN, else NULL */
    int *worker_model_ids;    /* per-engine model id after the train phase */
} PipelineSession;


Datum nr_pipeline_init(PG_FUNCTION_ARGS);
Datum nr_pipeline_push_slot(PG_FUNCTION_ARGS);
Datum nr_pipeline_state_change(PG_FUNCTION_ARGS);
Datum nr_pipeline_close(PG_FUNCTION_ARGS);

#endif //INTERFACE2_H
