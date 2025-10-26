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
    int epoch;
    int nfeat;
    PredictType type;

    int n_features;
    char **feature_names;
    char *target;
    TupleDesc tupdesc;

    // number of batches for train, eval, test
    int nb_tr, nb_ev, nb_te;

    // batch data
    HeapTuple *batch_vals;
    int batch_count;
    int batch_capacity;

    HTAB *class_id_map;
    List *id_class_map;
} PipelineSession;


Datum nr_pipeline_init(PG_FUNCTION_ARGS);
Datum nr_pipeline_push_slot(PG_FUNCTION_ARGS);
Datum nr_pipeline_state_change(PG_FUNCTION_ARGS);
Datum nr_pipeline_close(PG_FUNCTION_ARGS);

#endif //INTERFACE2_H
