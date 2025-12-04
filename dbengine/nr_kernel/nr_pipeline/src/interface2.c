/*-------------------------------------------------------------------------
 *
 * interface2.c
 *	  new interface (tuple-level) implementation for nr_pipeline
 *
 * ORIGINAL AUTHOR: Siqi Xiang
 *
 *-------------------------------------------------------------------------
 */
#include "interface2.h"

#include <utils/builtins.h>
#include <utils/array.h>
#include <utils/hsearch.h>
#include <utils/memutils.h>

#include <neurdb/predict.h>

#include <math.h>

#include "labeling/encode.h"
#include "utils/hash/md5.h"
#include "utils/network/task.h"


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(nr_pipeline_init);

PG_FUNCTION_INFO_V1(nr_pipeline_push_slot);

PG_FUNCTION_INFO_V1(nr_pipeline_state_change);

PG_FUNCTION_INFO_V1(nr_pipeline_close);


static void pipeline_close();


static PipelineSession PIPELINE_SESSION;

char *NrAIEngineHost = "localhost";
int NrAIEnginePort = 8090;

HTAB *last_class_id_map;
List *last_id_class_map;

typedef struct ClassIdHashEntry
{
    char *key;     /* must be first for HASH_STRINGS */
    int   id;
} ClassIdHashEntry;

// ------------------------ Util Functions ------------------------

static void make_class_id_map(
    const char *table_name,
    const char *label_col_name,
    HTAB **class_id_map,
    List **id_class_map
) {
    MemoryContext oldcxt;
    HASHCTL ctl;
    memset(&ctl, 0, sizeof(ctl));
    ctl.keysize = sizeof(char *);
    ctl.entrysize = sizeof(ClassIdHashEntry);
    
    PIPELINE_SESSION.hash_ctx = AllocSetContextCreate(
        TopMemoryContext,
        "neurdb class map ctx",
        ALLOCSET_DEFAULT_SIZES
    );
    oldcxt = MemoryContextSwitchTo(PIPELINE_SESSION.hash_ctx);

    HTAB *cimap = hash_create(
        "neurdb class id map",
        1024,
        &ctl,
        HASH_ELEM | HASH_STRINGS
    );
    List *icmap = NIL;

    MemoryContextSwitchTo(oldcxt);

    bool found = 0;

    SPI_connect();

    StringInfoData query;
    initStringInfo(&query);

    appendStringInfo(
        &query,
        "SELECT DISTINCT %s FROM %s ORDER BY %s ASC",
        label_col_name,
        table_name,
        label_col_name
    );
    SPI_execute(query.data, true, 0);

    if (SPI_processed > 0) {
        for (int i = 0; i < SPI_processed; i++) {
            char *label = SPI_getvalue(
                SPI_tuptable->vals[i],
                SPI_tuptable->tupdesc,
                1
            );
            oldcxt = MemoryContextSwitchTo(PIPELINE_SESSION.hash_ctx);
            char *label_copy = pstrdup(label);
            ClassIdHashEntry *entry = hash_search(cimap, (void *) label_copy, HASH_ENTER, &found);
            if (!found) {
                entry->id = i;
            } else {
                /* free label_copy because key already existed */
                pfree(label_copy);
            }
            icmap = lappend(icmap, makeString(pstrdup(label)));
            MemoryContextSwitchTo(oldcxt);
        }
    }
    *class_id_map = cimap;
    *id_class_map = icmap;

    SPI_finish();
}

static char *char_array2str(char **char_array, int n_elements) {
    StringInfoData str;
    initStringInfo(&str);
    for (int i = 0; i < n_elements; i++) {
        appendStringInfo(&str, "%s", char_array[i]);
        if (i < n_elements - 1) {
            appendStringInfoString(&str, ",");
        }
    }
    return str.data;
}

static void build_libsvm_data(
    SPITupleTable *tuptable,
    TupleDesc tupdesc,
    int n_features,
    char **feature_names,
    char *table_name,
    StringInfo libsvm_data,
    bool has_label,
    int label_col,
    const char *model_name
) {
    StringInfoData row_data;
    initStringInfo(&row_data);

    bool is_null;
    for (int i = 0; i < SPI_processed; i++) {
        resetStringInfo(&row_data);
        // handle label if present
        if (has_label) {
            Datum value = SPI_getbinval(
                tuptable->vals[i],
                tupdesc,
                label_col,
                &is_null
            );
            appendStringInfo(&row_data, "%d", DatumGetInt32(value));
        } else {
            appendStringInfoString(&row_data, "0"); // Default for inference
        }

        // process features
        for (int col = 0; col < n_features; col++) {
            Datum value = SPI_getbinval(
                tuptable->vals[i],
                tupdesc,
                col + 1,
                &is_null
            );
            int type = SPI_gettypeid(tupdesc, col + 1);
            switch (type) {
                case INT2OID:
                    appendStringInfo(&row_data, " %hd", DatumGetInt16(value));
                    break;
                case INT4OID:
                    appendStringInfo(&row_data, " %d", DatumGetInt32(value));
                    break;
                case INT8OID:
                    appendStringInfo(&row_data, " %ld", DatumGetInt64(value));
                    break;
                case FLOAT4OID:
                    appendStringInfo(&row_data, " %f", DatumGetFloat4(value));
                    break;
                case FLOAT8OID:
                    appendStringInfo(&row_data, " %lf", DatumGetFloat8(value));
                    break;
                case TEXTOID:
                case VARCHAROID:
                case CHAROID:
                    char *text = DatumGetCString(value);
                    if (strcmp(model_name, "auto_pipeline") != 0) {
                        int token = encode_text(text, table_name, feature_names[col]);
                        appendStringInfo(&row_data, " %d", token);
                    } else {
                        appendStringInfo(&row_data, " \"%s\"", text);
                    }
                    break;
                default:
                    elog(ERROR, "Unsupported data type");
            }
        }
        appendStringInfoString(&row_data, "\n");
        appendStringInfoString(libsvm_data, row_data.data);
    }
    pfree(row_data.data);
}

// ------------------------ Helper Functions ------------------------

static NrWebsocket *
connect_to_ai_engine() {
    NrWebsocket *ws = nws_initialize(NrAIEngineHost, NrAIEnginePort, "/ws", 10);
    nws_connect(ws);
    return ws;
}

static int
_lookup_model(const char *table_name, char **feature_names, int n_features, const char *target) {
    char *hash_features = nr_md5_list(feature_names, n_features);
    char *hash_target = nr_md5_str(target);

    StringInfoData query;
    initStringInfo(&query);
    appendStringInfo(
        &query,
        "SELECT model_id FROM router WHERE table_name = '%s' "
        "AND feature_columns = '%s' AND target_columns = '%s' "
        "LIMIT 1",
        table_name,
        hash_features,
        hash_target
    );

    int model_id = 0;
    SPI_connect();
    SPI_execute(query.data, true, 1);

    if (SPI_processed > 0) {
        bool is_null;
        Datum model_datum_id = SPI_getbinval(
            SPI_tuptable->vals[0],
            SPI_tuptable->tupdesc,
            1,
            &is_null
        );
        if (!is_null) {
            model_id = DatumGetInt32(model_datum_id);
        }
    }
    SPI_finish();
    return model_id;
}

static void
add_slot_to_batch(PipelineSession *session, TupleTableSlot *slot) {
    if (session->batch_vals == NULL) {
        session->batch_capacity = (session->batch_size > 0 ? session->batch_size : 1);
        // default to 1 if batch_size is 0
        session->batch_vals = (HeapTuple *) MemoryContextAlloc(
            TopMemoryContext,
            sizeof(HeapTuple) * session->batch_capacity
        );
        session->batch_count = 0;
    }
    if (session->batch_count == session->batch_capacity) {
        // TODO: ideally once we reach capacity we should run inference/training immediately, so (current implementation) there is no need to expand capacity
        // might have to expand capacity
        // session->batch_capacity *= 2;
        // session->batch_vals = (HeapTuple*) repalloc(session->batch_vals, sizeof(HeapTuple)*session->batch_capacity);
    }

    HeapTuple tuple = ExecCopySlotHeapTuple(slot); // copy to avoid dangling pointer
    session->batch_vals[session->batch_count++] = tuple;
}

static char *
run_infer_batch(PipelineSession *session, bool flush) {
    if (session->batch_capacity > session->batch_count && !flush) {
        // not enough data to run inference
        return NULL;
    }

    NrWebsocket *ws = session->ws;

    SPI_connect();

    SPITupleTable fake_table = {0};
    fake_table.tupdesc = session->tupdesc;
    fake_table.vals = session->batch_vals;
    extern uint64 SPI_processed;
    SPI_processed = (uint64) session->batch_count;

    StringInfoData libsvm;
    initStringInfo(&libsvm);
    build_libsvm_data(
        &fake_table,
        session->tupdesc,
        session->n_features,
        session->feature_names,
        session->table_name,
        &libsvm,
        false,
        0,
        session->model_name
    );
    nws_send_batch_data(ws, 0, S_INFERENCE, libsvm.data);
    nws_wait_completion(ws);
    /* reset completion flag */
    ws->completed = 0;
    char* payload = pstrdup(ws->result);

    SPI_finish();

    // clean up
    free(ws->result);
    for (int i = 0; i < session->batch_count; i++) {
        heap_freetuple(session->batch_vals[i]);
    }
    session->batch_count = 0;
    return payload;
}

static void
run_train_batch(PipelineSession *session, bool flush) {
    if (session->batch_capacity > session->batch_count && !flush) {
        // not enough data to run training
        return;
    }

    SPITupleTable fake_table = {0};
    fake_table.tupdesc = session->tupdesc;
    fake_table.vals = session->batch_vals;
    extern uint64 SPI_processed;
    SPI_processed = (uint64)session->batch_count;

    StringInfoData libsvm;
    initStringInfo(&libsvm);
    build_libsvm_data(
        &fake_table,
        session->tupdesc,
        session->n_features,
        session->feature_names,
        session->table_name,
        &libsvm,
        true,
        session->n_features+1,
        session->model_name
    );

    nws_send_batch_data(session->ws, 0, S_TRAIN, libsvm.data);

    for (int i=0;i<session->batch_count;i++) {
        heap_freetuple(session->batch_vals[i]);
    }
    session->batch_count = 0;
    resetStringInfo(&libsvm);
}

/* return true if model is found, false otherwise */
static bool
pipeline_init(
    const char *model_name,
    const char *table_name,
    int batch_size,
    int n_batches,
    int nfeat,
    char **feature_names,
    int n_features,
    const char *target,
    PredictType type,
    TupleDesc tupdesc
) {
    pipeline_close(); // clean up previous session if any

    PIPELINE_SESSION.state = PS_UNINIT;
    PIPELINE_SESSION.model_name = MemoryContextStrdup(TopMemoryContext, model_name);
    PIPELINE_SESSION.table_name = MemoryContextStrdup(TopMemoryContext, table_name);
    PIPELINE_SESSION.batch_size = batch_size;
    // PIPELINE_SESSION.epoch = epoch;
    PIPELINE_SESSION.n_batches = n_batches;
    PIPELINE_SESSION.nfeat = nfeat;
    PIPELINE_SESSION.type = type;
    PIPELINE_SESSION.tupdesc = tupdesc;
    PIPELINE_SESSION.n_features = n_features;
    PIPELINE_SESSION.feature_names = (char **) MemoryContextAlloc(TopMemoryContext, sizeof(char *) * n_features);
    for (int i = 0; i < n_features; i++) {
        PIPELINE_SESSION.feature_names[i] = MemoryContextStrdup(TopMemoryContext, feature_names[i]);
    }
    PIPELINE_SESSION.target = MemoryContextStrdup(TopMemoryContext, target);

    // look up for existing model
    int model_id = _lookup_model(PIPELINE_SESSION.table_name, PIPELINE_SESSION.feature_names,
                                PIPELINE_SESSION.n_features, PIPELINE_SESSION.target);
    if (model_id > 0) {
        // model found -> inference mode
        PIPELINE_SESSION.model_id = model_id;
        if (PIPELINE_SESSION.type == PREDICT_CLASS) {
            if (last_class_id_map) {
                hash_destroy(last_class_id_map);
                last_class_id_map = NULL;
            }
            if (last_id_class_map) {
                list_free_deep(last_id_class_map);
                last_id_class_map = NIL;
            }
            make_class_id_map(PIPELINE_SESSION.table_name, PIPELINE_SESSION.target, &last_class_id_map,
                              &last_id_class_map);
            PIPELINE_SESSION.class_id_map = last_class_id_map;
            PIPELINE_SESSION.id_class_map = last_id_class_map;
        }
        PIPELINE_SESSION.state = PS_INFER;
        PIPELINE_SESSION.ws = connect_to_ai_engine();

        int n_class = (PIPELINE_SESSION.type == PREDICT_CLASS && PIPELINE_SESSION.class_id_map)
                          ? hash_get_num_entries(PIPELINE_SESSION.class_id_map)
                          : -1;

        // send inference task to AI engine
        InferenceTaskSpec *it = malloc(sizeof(InferenceTaskSpec));
        init_inference_task_spec(
            it,
            PIPELINE_SESSION.model_name,
            PIPELINE_SESSION.batch_size,
            PIPELINE_SESSION.n_batches,
            "metrics",
            80,
            PIPELINE_SESSION.nfeat,
            PIPELINE_SESSION.n_features,
            n_class,
            PIPELINE_SESSION.model_id,
            char_array2str(PIPELINE_SESSION.feature_names, PIPELINE_SESSION.n_features),
            PIPELINE_SESSION.target
        );
        nws_send_task(PIPELINE_SESSION.ws, T_INFERENCE, PIPELINE_SESSION.table_name, it);
        free_inference_task_spec(it);

        return true;
    } else {
        // model not found -> training mode
        /*
         * TODO: we leave nb_tr to PIPELINE_SESSION.n_batches and other to 0 for now
         * meaning using all batches for training and no eval/test
         */
        PIPELINE_SESSION.nb_tr = PIPELINE_SESSION.n_batches;
        PIPELINE_SESSION.nb_ev = 0;
        PIPELINE_SESSION.nb_te = 0;

        if (PIPELINE_SESSION.type == PREDICT_CLASS) {
            if (last_class_id_map) {
                hash_destroy(last_class_id_map);
                last_class_id_map = NULL;
            }
            if (last_id_class_map) {
                list_free_deep(last_id_class_map);
                last_id_class_map = NIL;
            }
            make_class_id_map(
                PIPELINE_SESSION.table_name,
                PIPELINE_SESSION.target,
                &last_class_id_map,
                &last_id_class_map
            );
            PIPELINE_SESSION.class_id_map = last_class_id_map;
            PIPELINE_SESSION.id_class_map = last_id_class_map;
        }

        PIPELINE_SESSION.ws = connect_to_ai_engine();
        int n_class = (PIPELINE_SESSION.type == PREDICT_CLASS && PIPELINE_SESSION.class_id_map)
                        ? hash_get_num_entries(PIPELINE_SESSION.class_id_map)
                        : -1;

        // send training task to AI engine
        TrainTaskSpec *tt = malloc(sizeof(TrainTaskSpec));
        // TODO: we pass 0 for nb_tr/nb_ev/nb_te for now, need to fix
        init_train_task_spec(
            tt,
            PIPELINE_SESSION.model_name,
            PIPELINE_SESSION.batch_size,
            1,
            PIPELINE_SESSION.nb_tr,
            PIPELINE_SESSION.nb_ev,
            PIPELINE_SESSION.nb_te,
            0.001,
            "optimizer",
            "loss",
            "metrics",
            80,
            char_array2str(PIPELINE_SESSION.feature_names, PIPELINE_SESSION.n_features),
            PIPELINE_SESSION.target,
            PIPELINE_SESSION.nfeat,
            PIPELINE_SESSION.n_features,
            n_class
        );
        nws_send_task(PIPELINE_SESSION.ws, T_TRAIN, PIPELINE_SESSION.table_name, tt);
        free_train_task_spec(tt);
        // set to training state
        PIPELINE_SESSION.state = PS_TRAIN;

        return false;
    }
}

static bool
pipeline_push_slot(TupleTableSlot **slot, int num_slot, char **infer_result_out, bool flush) {
    if (PIPELINE_SESSION.state == PS_UNINIT) {
        elog(ERROR, "nr_state not initialized, please call pipeline_init first");
    }
    for (int i = 0; i < num_slot; i++) {
        add_slot_to_batch(&PIPELINE_SESSION, slot[i]);
    }
    if (PIPELINE_SESSION.batch_count < PIPELINE_SESSION.batch_size && !flush) {
        if (infer_result_out) *infer_result_out = NULL;
        return false; // not enough data yet
    }

    if (PIPELINE_SESSION.state == PS_TRAIN) {
        run_train_batch(&PIPELINE_SESSION, flush);
        if (infer_result_out) {
            // no inference result during training
            *infer_result_out = NULL;
        }
        return true;
    } else {
        // PS_INFER
        char *res = run_infer_batch(&PIPELINE_SESSION, flush);
        if (infer_result_out) {
            *infer_result_out = res;
        }
        return true;
    }
}

static void
_clean_up_conn(NrWebsocket *ws) {
  // close the connection
  nws_disconnect(ws);
  nws_free_websocket(ws);
}

static void
pipeline_state_change(bool to_inference) {
    if (to_inference) {
        // TRAIN -> INFER
        run_train_batch(&PIPELINE_SESSION, /*flush=*/true);
        nws_wait_completion(PIPELINE_SESSION.ws);
        PIPELINE_SESSION.model_id = PIPELINE_SESSION.ws->model_id;

        /* reset websocket with a new connection */
        _clean_up_conn(PIPELINE_SESSION.ws);
        PIPELINE_SESSION.ws = connect_to_ai_engine();

        int n_class = (PIPELINE_SESSION.type == PREDICT_CLASS && PIPELINE_SESSION.class_id_map)
                        ? hash_get_num_entries(PIPELINE_SESSION.class_id_map) : -1;
        InferenceTaskSpec *it = malloc(sizeof(InferenceTaskSpec));
        init_inference_task_spec(
            it,
            PIPELINE_SESSION.model_name,
            PIPELINE_SESSION.batch_size,
            PIPELINE_SESSION.n_batches,
            "metrics",
            80,
            PIPELINE_SESSION.nfeat,
            PIPELINE_SESSION.n_features,
            n_class,
            PIPELINE_SESSION.model_id,
            char_array2str(
                PIPELINE_SESSION.feature_names,
                PIPELINE_SESSION.n_features
            ),
            PIPELINE_SESSION.target
        );
        nws_send_task(PIPELINE_SESSION.ws, T_INFERENCE, PIPELINE_SESSION.table_name, it);
        free_inference_task_spec(it);

        PIPELINE_SESSION.state = PS_INFER;
    } else {
        // INFER -> TRAIN
        // TODO: not supported because ideally there is no INFER -> TRAIN transition
        elog(ERROR, "INFER -> TRAIN state change is not supported");
    }
}

static void
pipeline_close() {
    if (PIPELINE_SESSION.ws) {
        _clean_up_conn(PIPELINE_SESSION.ws);
        PIPELINE_SESSION.ws = NULL;
    }

    if (PIPELINE_SESSION.batch_vals) {
        for (int i = 0; i < PIPELINE_SESSION.batch_count; i++) {
            heap_freetuple(PIPELINE_SESSION.batch_vals[i]);
        }
        pfree(PIPELINE_SESSION.batch_vals);
    }

    if (PIPELINE_SESSION.feature_names) {
        for (int i = 0; i < PIPELINE_SESSION.n_features; i++) pfree(PIPELINE_SESSION.feature_names[i]);
        pfree(PIPELINE_SESSION.feature_names);
    }
    if (PIPELINE_SESSION.target) pfree(PIPELINE_SESSION.target);
    if (PIPELINE_SESSION.model_name) pfree(PIPELINE_SESSION.model_name);
    if (PIPELINE_SESSION.table_name) pfree(PIPELINE_SESSION.table_name);

    if (PIPELINE_SESSION.hash_ctx) {
        MemoryContextDelete(PIPELINE_SESSION.hash_ctx);
        PIPELINE_SESSION.hash_ctx = NULL;
    }

    memset(&PIPELINE_SESSION, 0, sizeof(PIPELINE_SESSION));

}

static char **
_array_to_cstring_list(ArrayType *arr, int *out_nelems)
{
    Datum *elems;
    bool *nulls;
    int nelems;

    deconstruct_array(arr,
                      TEXTOID,   /* element type */
                      -1,        /* typlen for text */
                      false,     /* typbyval */
                      'i',       /* typalign */
                      &elems,
                      &nulls,
                      &nelems);

    char **result = NULL;
    if (nelems > 0)
    {
        result = (char **) palloc(sizeof(char *) * nelems);
        for (int i = 0; i < nelems; i++)
        {
            if (nulls[i])
                result[i] = NULL;
            else
                result[i] = TextDatumGetCString(elems[i]);
        }
    }

    pfree(elems);
    pfree(nulls);

    if (out_nelems)
        *out_nelems = nelems;

    return result;
}

Datum
nr_pipeline_init(PG_FUNCTION_ARGS) {
    char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));
    char *table_name = text_to_cstring(PG_GETARG_TEXT_P(1));
    int batch_size = PG_GETARG_INT32(2);
    int n_batches = PG_GETARG_INT32(3);
    int nfeat = PG_GETARG_INT32(4);

    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(5);
    int n_features = 0;
    char **feature_names = _array_to_cstring_list(arr, &n_features);

    char *target = text_to_cstring(PG_GETARG_TEXT_P(6));
    PredictType type = PG_GETARG_INT32(7);
    TupleDesc tupdesc = (TupleDesc) PG_GETARG_DATUM(8);

    bool is_inference = pipeline_init(
        model_name,
        table_name,
        batch_size,
        n_batches,
        nfeat,
        feature_names,
        n_features,
        target,
        type,
        tupdesc
    );
    PG_RETURN_BOOL(is_inference);
}

Datum
nr_pipeline_push_slot(PG_FUNCTION_ARGS) {
    TupleTableSlot **slot = (TupleTableSlot **) PG_GETARG_POINTER(0);
    int num_slot = PG_GETARG_INT32(1);
    bool flush = PG_GETARG_BOOL(2);

    char **infer_result_out = (char **) palloc(sizeof(char *));
    pipeline_push_slot(slot, num_slot, infer_result_out, flush);

    NeurDBInferenceResult *result = palloc(sizeof(NeurDBInferenceResult));
    // TODO: infer the type of the result
    if (PIPELINE_SESSION.type == PREDICT_CLASS) {
        result->typeoid = TEXTOID;
    } else if (PIPELINE_SESSION.type == PREDICT_VALUE) {
        result->typeoid = FLOAT8OID;
    } else {
        elog(ERROR, "Unsupported data type");
    }
    result->result = *infer_result_out;
    result->id_class_map = last_id_class_map;

    PG_RETURN_POINTER(result);
}

Datum
nr_pipeline_state_change(PG_FUNCTION_ARGS) {
    bool to_inference = PG_GETARG_BOOL(0);
    pipeline_state_change(to_inference);
    PG_RETURN_VOID();
}

Datum
nr_pipeline_close(PG_FUNCTION_ARGS) {
    pipeline_close();
    PG_RETURN_VOID();
}
