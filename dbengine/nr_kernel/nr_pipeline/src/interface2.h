/*-------------------------------------------------------------------------
 *
 * interface2.c
 *	  new interface (tuple-level) implementation for nr_pipeline
 *
 * ORIGINAL AUTHOR: Siqi Xiang
 *
 *-------------------------------------------------------------------------
 */
#ifndef NEW_INTERFACE_H
#define NEW_INTERFACE_H

#include <postgres.h>
#include <executor/spi.h>

#include "utils/network/websocket.h"


void pipeline_init(
    const char *model_name,
    const char *table_name,
    int batch_size,
    int epoch,
    int nfeat,
    char **feature_names,
    int n_features,
    const char *target,
    PredictType type,
    TupleDesc tupdesc
);

bool pipeline_push_slot(TupleTableSlot *slot, char **infer_result_out, bool flush);

void pipeline_state_change(bool to_inference);

void pipeline_close();

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

#endif //NEW_INTERFACE_H
