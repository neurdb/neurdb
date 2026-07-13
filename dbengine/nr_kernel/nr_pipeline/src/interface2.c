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

#include <access/relation.h>
#include <access/heapam.h>
#include <access/htup.h>
#include <access/table.h>
#include <access/genam.h>
#include <catalog/namespace.h>
#include <catalog/indexing.h>
#include <catalog/nr_aiengine.h>
#include <nodes/makefuncs.h>
#include <utils/builtins.h>
#include <utils/array.h>
#include <utils/hsearch.h>
#include <utils/memutils.h>

#include <neurdb/predict.h>

#include <math.h>

#include <utils/guc.h>
#include <limits.h>

#include "labeling/encode.h"
#include "utils/hash/md5.h"
#include "utils/network/task.h"


PG_MODULE_MAGIC;

void _PG_init(void);

/*
 * nr_pipeline.engine_pin: route this session's PREDICT to a single AI engine
 * (index into nr_aiengine catalog order, modulo the engine count) instead of
 * broadcasting the in-context train phase to every engine and sharding
 * inference across all of them.  -1 (default) keeps broadcast+shard.
 *
 * Rationale: for an in-context model the per-task context fit is DUPLICATED
 * on every engine under broadcast, so running many concurrent PREDICT tasks
 * against the full pool makes each server fit every task's context and
 * task-level parallelism does not scale.  Pinning gives each concurrent task
 * its own engine: fits spread across the pool instead of being replicated.
 */
static int nr_engine_pin = -1;

void
_PG_init(void)
{
	DefineCustomIntVariable("nr_pipeline.engine_pin",
							"Pin this session's PREDICT to one AI engine "
							"(index into nr_aiengine; -1 = use all engines).",
							NULL,
							&nr_engine_pin,
							-1, -1, INT_MAX,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);
}

PG_FUNCTION_INFO_V1(nr_pipeline_init);

PG_FUNCTION_INFO_V1(nr_pipeline_push_slot);

PG_FUNCTION_INFO_V1(nr_pipeline_state_change);

PG_FUNCTION_INFO_V1(nr_pipeline_close);


static void pipeline_close();

/* forward declarations (used by send_inference_task before their definitions) */
static bool _is_tabpfn(const char *model_name);
static char *_build_col_types(TupleDesc tupdesc, int n_features);


static PipelineSession PIPELINE_SESSION;

HTAB *last_class_id_map;
List *last_id_class_map;

typedef struct ClassIdHashEntry
{
    char *key;     /* must be first for HASH_STRINGS */
    int   id;
} ClassIdHashEntry;

typedef struct EngineEndpoint {
    char *host;
    int port;
} EngineEndpoint;

typedef struct InferJob {
    int job_id;
    char *payload;
} InferJob;

typedef struct InferResult {
    int job_id;
    char *payload;
} InferResult;

// ------------------------ Queue Structures ------------------------

typedef struct InferJobNode {
    InferJob job;
    struct InferJobNode *next;
} InferJobNode;

typedef struct InferResultNode {
    InferResult result;
    struct InferResultNode *next;
} InferResultNode;

typedef struct InferJobQueue {
    InferJobNode *head;
    InferJobNode *tail;
    size_t size;
    bool closed;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} InferJobQueue;

typedef struct InferResultQueue {
    InferResultNode *head;
    InferResultNode *tail;
    size_t size;
    bool closed;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} InferResultQueue;

typedef struct InferResultState {
    int active;
    int expected;
    int collected;
    int capacity;
    char **results;
    pthread_mutex_t mutex;
    pthread_cond_t active_cond;
    pthread_cond_t done_cond;
} InferResultState;

typedef struct DistributedInfer DistributedInfer;

/*-------------------------------------------------------------
 * |                     DistributedInfer                     |
 * | InferWorker[0] | InferWorker[1] | ... | InferWorker[N-1] |
 * |                      InferCollector                      |
 *-------------------------------------------------------------*/

typedef struct InferWorker {
    pthread_t thread;
    NrWebsocket *ws;
    DistributedInfer *dist;
} InferWorker;

struct DistributedInfer {
    int worker_count;
    InferWorker *workers;
    InferJobQueue job_queue;
    InferResultQueue result_queue;
    pthread_t collector_thread;
    InferResultState result_state;
    int shutting_down;
};


// ------------------------ Util Functions ------------------------

static void
init_infer_job_queue(InferJobQueue *queue) {
    memset(queue, 0, sizeof(*queue));
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
}

static void
destroy_infer_job_queue(InferJobQueue *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->closed = true;
    InferJobNode *node = queue->head;
    while (node) {
        InferJobNode *next = node->next;
        free(node->job.payload);
        free(node);
        node = next;
    }
    queue->head = queue->tail = NULL;
    queue->size = 0;
    pthread_mutex_unlock(&queue->mutex);
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);
}

static void
close_infer_job_queue(InferJobQueue *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->closed = true;
    pthread_cond_broadcast(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

static bool
enqueue_infer_job(InferJobQueue *queue, InferJob job) {
    InferJobNode *node = (InferJobNode *) malloc(sizeof(*node));
    if (!node) {
        return false;
    }
    node->job = job;
    node->next = NULL;

    pthread_mutex_lock(&queue->mutex);
    if (queue->closed) {
        // free resources
        pthread_mutex_unlock(&queue->mutex);
        free(node->job.payload);
        free(node);
        return false;
    }
    if (queue->tail) {
        queue->tail->next = node;
    } else {
        queue->head = node;
    }
    queue->tail = node;
    queue->size++;
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

static bool
dequeue_infer_job(InferJobQueue *queue, InferJob *out) {
    pthread_mutex_lock(&queue->mutex);
    while (!queue->head && !queue->closed) {
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }
    if (!queue->head) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    InferJobNode *node = queue->head;
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    queue->size--;
    *out = node->job;
    free(node);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

static void
init_infer_result_queue(InferResultQueue *queue) {
    memset(queue, 0, sizeof(*queue));
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
}

static void
destroy_infer_result_queue(InferResultQueue *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->closed = true;
    InferResultNode *node = queue->head;
    while (node) {
        InferResultNode *next = node->next;
        free(node->result.payload);
        free(node);
        node = next;
    }
    queue->head = queue->tail = NULL;
    queue->size = 0;
    pthread_mutex_unlock(&queue->mutex);
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);
}

static void
close_infer_result_queue(InferResultQueue *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->closed = true;
    pthread_cond_broadcast(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

static bool
enqueue_infer_result(InferResultQueue *queue, InferResult result) {
    InferResultNode *node = (InferResultNode *) malloc(sizeof(*node));
    if (!node) {
        return false;
    }
    node->result = result;
    node->next = NULL;

    pthread_mutex_lock(&queue->mutex);
    if (queue->closed) {
        pthread_mutex_unlock(&queue->mutex);
        free(node->result.payload);
        free(node);
        return false;
    }
    if (queue->tail) {
        queue->tail->next = node;
    } else {
        queue->head = node;
    }
    queue->tail = node;
    queue->size++;
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

static bool
dequeue_infer_result(InferResultQueue *queue, InferResult *out) {
    pthread_mutex_lock(&queue->mutex);
    while (!queue->head && !queue->closed) {
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }
    if (!queue->head) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    InferResultNode *node = queue->head;
    queue->head = node->next;
    if (!queue->head) {
        queue->tail = NULL;
    }
    queue->size--;
    *out = node->result;
    free(node);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

static void
init_result_state(InferResultState *state) {
    memset(state, 0, sizeof(*state));
    pthread_mutex_init(&state->mutex, NULL);
    pthread_cond_init(&state->active_cond, NULL);
    pthread_cond_init(&state->done_cond, NULL);
}

static void
destroy_result_state(InferResultState *state) {
    if (state->results) {
        for (int i = 0; i < state->capacity; i++) {
            free(state->results[i]);
        }
        free(state->results);
        state->results = NULL;
    }
    pthread_mutex_destroy(&state->mutex);
    pthread_cond_destroy(&state->active_cond);
    pthread_cond_destroy(&state->done_cond);
}

static void
set_result_state_active(InferResultState *state, int expected, int capacity) {
    pthread_mutex_lock(&state->mutex);
    if (state->results) {
        for (int i = 0; i < state->capacity; i++) {
            free(state->results[i]);
        }
        free(state->results);
    }
    state->capacity = capacity;
    state->expected = expected;
    state->collected = 0;
    state->results = (char **) calloc(capacity, sizeof(char *));
    state->active = 1;
    pthread_cond_broadcast(&state->active_cond);
    pthread_mutex_unlock(&state->mutex);
}

static void
wait_result_state_done(InferResultState *state) {
    pthread_mutex_lock(&state->mutex);
    while (state->active && state->collected < state->expected) {
        pthread_cond_wait(&state->done_cond, &state->mutex);
    }
    pthread_mutex_unlock(&state->mutex);
}

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

static EngineEndpoint *
load_ai_engines(int *out_count) {
    int ret = SPI_connect();
    if (ret != SPI_OK_CONNECT) {
        elog(ERROR, "SPI_connect failed: %d", ret);
    }

    ret = SPI_execute(
        "SELECT aieaddr, aieport "
        "FROM pg_catalog.nr_aiengine",
        true,
        0
    );

    if (ret != SPI_OK_SELECT || SPI_processed == 0) {
        SPI_finish();
        elog(ERROR, "nr_aiengine catalog is empty");
    }

    // allocate endpoints
    int count = (int) SPI_processed;
    EngineEndpoint *endpoints = (EngineEndpoint *) malloc(sizeof(*endpoints) * count);
    if (!endpoints) {
        SPI_finish();
        elog(ERROR, "failed to allocate ai engine endpoints");
    }
    memset(endpoints, 0, sizeof(*endpoints) * count);

    // construct AIEngine endpoints
    for (int i = 0; i < count; i++) {
        bool isnull = false;
        Datum addr_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull) {
            char *addr = text_to_cstring(DatumGetTextPP(addr_datum));
            endpoints[i].host = strdup(addr);
            pfree(addr);
        }

        Datum port_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &isnull);
        if (!isnull) {
            endpoints[i].port = DatumGetInt32(port_datum);
        }
    }
    SPI_finish();

    *out_count = count;
    return endpoints;
}

static void
free_ai_engines(EngineEndpoint *endpoints, int count) {
    if (!endpoints) {
        return;
    }
    for (int i = 0; i < count; i++) {
        free(endpoints[i].host);
    }
    free(endpoints);
}

static void
send_inference_task(NrWebsocket *ws, const PipelineSession *session, int model_id) {
    int n_class = (session->type == PREDICT_CLASS && session->class_id_map)
                      ? hash_get_num_entries(session->class_id_map)
                      : -1;
    InferenceTaskSpec *it = malloc(sizeof(InferenceTaskSpec));
    init_inference_task_spec(
        it,
        session->model_name,
        session->batch_size,
        session->n_batches,
        "metrics",
        80,
        session->nfeat,
        session->n_features,
        n_class,
        model_id,
        char_array2str(session->feature_names, session->n_features),
        session->target
    );
    if (_is_tabpfn(session->model_name)) {
        char *ct = _build_col_types(session->tupdesc, session->n_features);
        it->colTypes = strdup(ct);
        pfree(ct);
    }
    nws_send_task(ws, T_INFERENCE, session->table_name, it);
    free_inference_task_spec(it);
}

static void *
infer_worker_main(void *arg) {
    InferWorker *worker = (InferWorker *) arg;
    DistributedInfer *dist = worker->dist;

    for (;;) {
        InferJob job;
        if (!dequeue_infer_job(&dist->job_queue, &job)) {
            break;
        }
        nws_send_batch_data(worker->ws, job.job_id, S_INFERENCE, job.payload);
        nws_wait_completion(worker->ws);
        worker->ws->completed = 0;

        char *result_copy = NULL;
        if (worker->ws->result) {
            result_copy = strdup(worker->ws->result);
            free(worker->ws->result);
            worker->ws->result = NULL;
        }

        InferResult result = {
            .job_id = job.job_id,
            .payload = result_copy
        };
        enqueue_infer_result(&dist->result_queue, result);
        free(job.payload);
    }
    return NULL;
}

static void *
infer_collector_main(void *arg) {
    DistributedInfer *dist = (DistributedInfer *) arg;

    for (;;) {
        InferResult result;
        if (!dequeue_infer_result(&dist->result_queue, &result)) {
            break;
        }

        pthread_mutex_lock(&dist->result_state.mutex);
        while (!dist->result_state.active && !dist->shutting_down) {
            pthread_cond_wait(&dist->result_state.active_cond, &dist->result_state.mutex);
        }
        if (dist->shutting_down) {
            pthread_mutex_unlock(&dist->result_state.mutex);
            free(result.payload);
            break;
        }
        if (result.job_id >= 0 && result.job_id < dist->result_state.capacity) {
            dist->result_state.results[result.job_id] = result.payload;
            dist->result_state.collected++;
            if (dist->result_state.collected >= dist->result_state.expected) {
                dist->result_state.active = 0;
                pthread_cond_broadcast(&dist->result_state.done_cond);
            }
        } else {
            free(result.payload);
        }
        pthread_mutex_unlock(&dist->result_state.mutex);
    }
    return NULL;
}

/*
 * Spin up one worker (thread + websocket) per AI engine and send each the
 * inference task.  When worker_model_ids is non-NULL it carries one model id
 * per engine (broadcast-trained in-context models such as tabpfn, whose model
 * ids are AI-server-process-local); otherwise session->model_id is shared by
 * every worker (models persisted in the DB-backed model repo).
 *
 * Endpoints captured in the session at train time take priority over a fresh
 * catalog load so the worker order always matches worker_model_ids.
 */
static DistributedInfer *
distributed_infer_create(const PipelineSession *session, const int *worker_model_ids) {
    int engine_count = 0;
    EngineEndpoint *engines = NULL;

    if (session->eng_count > 0) {
        engine_count = session->eng_count;
        engines = (EngineEndpoint *) malloc(sizeof(*engines) * engine_count);
        if (!engines) {
            elog(ERROR, "failed to allocate ai engine endpoints");
        }
        for (int i = 0; i < engine_count; i++) {
            engines[i].host = strdup(session->eng_hosts[i]);
            engines[i].port = session->eng_ports[i];
        }
    } else {
        engines = load_ai_engines(&engine_count);
    }
    if (engine_count <= 0) {
        free_ai_engines(engines, engine_count);
        return NULL;
    }

    DistributedInfer *dist = (DistributedInfer *) malloc(sizeof(*dist));
    if (!dist) {
        free_ai_engines(engines, engine_count);
        elog(ERROR, "failed to allocate distributed inference state");
    }
    memset(dist, 0, sizeof(*dist));

    init_infer_job_queue(&dist->job_queue);
    init_infer_result_queue(&dist->result_queue);
    init_result_state(&dist->result_state);

    dist->worker_count = engine_count;
    dist->workers = (InferWorker *) malloc(sizeof(*dist->workers) * engine_count);
    if (!dist->workers) {
        free_ai_engines(engines, engine_count);
        destroy_infer_job_queue(&dist->job_queue);
        destroy_infer_result_queue(&dist->result_queue);
        destroy_result_state(&dist->result_state);
        free(dist);
        elog(ERROR, "failed to allocate distributed inference workers");
    }
    memset(dist->workers, 0, sizeof(*dist->workers) * engine_count);

    for (int i = 0; i < engine_count; i++) {
        dist->workers[i].dist = dist;
        dist->workers[i].ws = nws_initialize(engines[i].host, engines[i].port, "/ws", 10);
        nws_connect(dist->workers[i].ws);
        send_inference_task(dist->workers[i].ws, session,
                            worker_model_ids ? worker_model_ids[i] : session->model_id);
    }

    for (int i = 0; i < engine_count; i++) {
        if (pthread_create(&dist->workers[i].thread, NULL, infer_worker_main, &dist->workers[i]) != 0) {
            elog(ERROR, "failed to create inference worker thread");
        }
    }

    if (pthread_create(&dist->collector_thread, NULL, infer_collector_main, dist) != 0) {
        elog(ERROR, "failed to create inference collector thread");
    }

    free_ai_engines(engines, engine_count);
    return dist;
}

static List *
get_primary_key_ids(const char *table_name) {
	RangeVar   *rv = makeRangeVar(NULL, table_name, -1);
	Oid			relid = RangeVarGetRelid(rv, NoLock, false);
	Relation	rel = relation_open(relid, AccessShareLock);
	Oid			ident_index;
	List	   *keys = NIL;

	ident_index = RelationGetReplicaIndex(rel);

	if (OidIsValid(ident_index))
	{
		Relation	idxrel;
		Form_pg_index idx;

		idxrel = index_open(ident_index, AccessShareLock);
		idx = idxrel->rd_index;

		for (int i = 0; i < idx->indnatts; i++)
		{
			if (idx->indkey.values[i] == 0)
				ereport(ERROR,
						(errmsg("replica identity index contains expressions")));

			keys = lappend_int(keys, idx->indkey.values[i] - 1);
		}

		index_close(idxrel, AccessShareLock);
	}

	relation_close(rel, AccessShareLock);

	return keys;
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
	const char *model_name,
	List *primary_key_ids
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
            int v = DatumGetInt32(value);
            appendStringInfo(&row_data, "%d", v);
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

			/* col because of the first column is the label */
			if (list_member_int(primary_key_ids, col)) {
				// skip primary key
				continue;
			}

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

static bool
_is_tabpfn(const char *model_name) {
    return model_name && strcmp(model_name, "tabpfn") == 0;
}

/* Replace framing chars inside a field so the tab/newline layout survives. */
static void
_sanitize_field(char *s) {
    for (char *p = s; *p; p++) {
        if (*p == '\t' || *p == '\n' || *p == '\r') {
            *p = ' ';
        }
    }
}

/*
 * Typed payload for in-context, type-aware models (e.g. tabpfn).
 *
 * One row per tuple, fields TAB-separated, label first then the features in
 * column order. Each value is the type's canonical text (via SPI_getvalue /
 * the type output function) so numbers, timestamps and text all survive
 * losslessly; a SQL NULL becomes an empty field. This is the dual of
 * build_libsvm_data, which forces everything to dense integers/floats and
 * tokenizes text -- lossy and wrong for tabpfn's type-aware preprocessing.
 */
static void
build_typed_data(
    SPITupleTable *tuptable,
    TupleDesc tupdesc,
    int n_features,
    StringInfo out,
    bool has_label,
    int label_col
) {
    StringInfoData row;
    initStringInfo(&row);
    for (int i = 0; i < SPI_processed; i++) {
        resetStringInfo(&row);
        // label occupies the first field (empty placeholder during inference)
        if (has_label) {
            char *lv = SPI_getvalue(tuptable->vals[i], tupdesc, label_col);
            if (lv) {
                _sanitize_field(lv);
                appendStringInfoString(&row, lv);
                pfree(lv);
            }
        }
        // features
        for (int col = 0; col < n_features; col++) {
            appendStringInfoChar(&row, '\t');
            char *v = SPI_getvalue(tuptable->vals[i], tupdesc, col + 1);
            if (v) {
                _sanitize_field(v);
                appendStringInfoString(&row, v);
                pfree(v);
            }
        }
        appendStringInfoChar(&row, '\n');
        appendStringInfoString(out, row.data);
    }
    pfree(row.data);
}

/*
 * Comma-separated Postgres type names for the first n_features columns (the
 * feature columns, same positional convention as build_*_data). Sent in the
 * task spec so the engine can derive type-aware stype hints (e.g. timestamp)
 * that it cannot recover from the stringified values alone. Returns palloc'd
 * memory.
 */
static char *
_build_col_types(TupleDesc tupdesc, int n_features) {
    StringInfoData s;
    initStringInfo(&s);
    for (int col = 0; col < n_features; col++) {
        if (col > 0) {
            appendStringInfoChar(&s, ',');
        }
        Oid t = SPI_gettypeid(tupdesc, col + 1);
        char *name = format_type_be(t);
        appendStringInfoString(&s, name);
        pfree(name);
    }
    return s.data;
}

// ------------------------ Helper Functions ------------------------

static NrWebsocket *
connect_to_ai_engine() {
    int engine_count = 0;
    EngineEndpoint *engines = load_ai_engines(&engine_count);
    if (engine_count <= 0) {
        free_ai_engines(engines, engine_count);
        elog(ERROR, "nr_aiengine catalog is empty");
    }

    elog(DEBUG1, "connecting to AI engine at: %s:%d", engines[0].host, engines[0].port);

    NrWebsocket *ws = nws_initialize(engines[0].host, engines[0].port, "/ws", 10);
    nws_connect(ws);

    free_ai_engines(engines, engine_count);

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

    SPI_connect();

    SPITupleTable fake_table = {0};
    fake_table.tupdesc = session->tupdesc;
    fake_table.vals = session->batch_vals;
    extern uint64 SPI_processed;
    SPI_processed = (uint64) session->batch_count;

    StringInfoData libsvm;
    initStringInfo(&libsvm);
    if (_is_tabpfn(session->model_name)) {
        build_typed_data(
            &fake_table,
            session->tupdesc,
            session->n_features,
            &libsvm,
            false,
            0
        );
    } else {
        build_libsvm_data(
            &fake_table,
            session->tupdesc,
            session->n_features,
            session->feature_names,
            session->table_name,
            &libsvm,
            false,
            0,
			session->model_name,
			session->primary_key_ids
        );
    }
    char *payload = NULL;
    if (session->dist_infer) {
        int total_lines = 0;
        bool in_line = false;
        for (const char *p = libsvm.data; *p; p++) {
            if (*p == '\n') {
                if (in_line) {
                    total_lines++;
                }
                in_line = false;
            } else {
                in_line = true;
            }
        }
        if (in_line) {
            total_lines++;
        }

        int worker_count = session->dist_infer->worker_count;
        if (total_lines > 0 && worker_count > 0) {
            int chunk_size = (total_lines + worker_count - 1) / worker_count;
            StringInfoData *chunks = (StringInfoData *) palloc(sizeof(*chunks) * worker_count);
            for (int i = 0; i < worker_count; i++) {
                initStringInfo(&chunks[i]);
            }

            const char *start = libsvm.data;
            int line_idx = 0;
            for (const char *p = libsvm.data; ; p++) {
                if (*p == '\n' || *p == '\0') {
                    size_t len = p - start;
                    if (len > 0) {
                        int chunk_idx = line_idx / chunk_size;
                        if (chunk_idx >= worker_count) {
                            chunk_idx = worker_count - 1;
                        }
                        appendBinaryStringInfo(&chunks[chunk_idx], start, len);
                        appendStringInfoChar(&chunks[chunk_idx], '\n');
                        line_idx++;
                    }
                    if (*p == '\0') {
                        break;
                    }
                    start = p + 1;
                }
            }

            int expected = 0;
            for (int i = 0; i < worker_count; i++) {
                if (chunks[i].len > 0) {
                    expected++;
                }
            }

            if (expected > 0) {
                set_result_state_active(&session->dist_infer->result_state, expected, worker_count);
                for (int i = 0; i < worker_count; i++) {
                    if (chunks[i].len == 0) {
                        continue;
                    }
                    InferJob job = {
                        .job_id = i,
                        .payload = strdup(chunks[i].data)
                    };
                    enqueue_infer_job(&session->dist_infer->job_queue, job);
                }

                wait_result_state_done(&session->dist_infer->result_state);

                StringInfoData combined;
                initStringInfo(&combined);
                pthread_mutex_lock(&session->dist_infer->result_state.mutex);
                for (int i = 0; i < session->dist_infer->result_state.capacity; i++) {
                    char *piece = session->dist_infer->result_state.results[i];
                    if (!piece || piece[0] == '\0') {
                        continue;
                    }
                    if (combined.len > 0 && combined.data[combined.len - 1] != ' ') {
                        appendStringInfoChar(&combined, ' ');
                    }
                    appendStringInfoString(&combined, piece);
                }
                for (int i = 0; i < session->dist_infer->result_state.capacity; i++) {
                    free(session->dist_infer->result_state.results[i]);
                    session->dist_infer->result_state.results[i] = NULL;
                }
                free(session->dist_infer->result_state.results);
                session->dist_infer->result_state.results = NULL;
                session->dist_infer->result_state.capacity = 0;
                session->dist_infer->result_state.expected = 0;
                session->dist_infer->result_state.collected = 0;
                pthread_mutex_unlock(&session->dist_infer->result_state.mutex);

                payload = pstrdup(combined.data);
                pfree(combined.data);
            }

            for (int i = 0; i < worker_count; i++) {
                pfree(chunks[i].data);
            }
            pfree(chunks);
        }
    } else {
        NrWebsocket *ws = session->ws;
        nws_send_batch_data(ws, 0, S_INFERENCE, libsvm.data);
        nws_wait_completion(ws);
        /* reset completion flag */
        ws->completed = 0;
        payload = pstrdup(ws->result);
        free(ws->result);
    }

    SPI_finish();

    // clean up
    for (int i = 0; i < session->batch_count; i++) {
        heap_freetuple(session->batch_vals[i]);
    }
    session->batch_count = 0;
    return payload;
}

static int
_label_col_id(TupleDesc tupdesc, const char *label_col) {
    // or directly inspect attributes:
    for (int i = 0; i < tupdesc->natts; i++)
    {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
        if (strncmp(attr->attname.data, label_col, strlen(label_col)) == 0) {
            return i + 1;
        }
    }

    return -1;
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
    if (_is_tabpfn(session->model_name)) {
        build_typed_data(
            &fake_table,
            session->tupdesc,
            session->n_features,
            &libsvm,
            true,
            session->label_col_id
        );
    } else {
        build_libsvm_data(
            &fake_table,
            session->tupdesc,
            session->n_features,
            session->feature_names,
            session->table_name,
            &libsvm,
            true,
            session->label_col_id,
			session->model_name,
			session->primary_key_ids
        );
    }

    if (session->train_wss) {
        /* broadcast the same context batch to every engine */
        for (int i = 0; i < session->eng_count; i++) {
            nws_send_batch_data(session->train_wss[i], 0, S_TRAIN, libsvm.data);
        }
    } else {
        nws_send_batch_data(session->ws, 0, S_TRAIN, libsvm.data);
    }

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
    int epoch,
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
    PIPELINE_SESSION.epoch = epoch;
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
    PIPELINE_SESSION.label_col_id = _label_col_id(tupdesc, target);
    PIPELINE_SESSION.dist_infer = NULL;
    if (PIPELINE_SESSION.label_col_id < 0) {
        elog(ERROR, "label column %s not found", target);
    }
	PIPELINE_SESSION.primary_key_ids = get_primary_key_ids(table_name);

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
        PIPELINE_SESSION.ws = NULL;
        PIPELINE_SESSION.dist_infer = distributed_infer_create(&PIPELINE_SESSION, NULL);
        if (!PIPELINE_SESSION.dist_infer) {
            PIPELINE_SESSION.ws = connect_to_ai_engine();
            send_inference_task(PIPELINE_SESSION.ws, &PIPELINE_SESSION, PIPELINE_SESSION.model_id);
        }

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

        /*
         * tabpfn is in-context with a process-local session store, so the
         * context (train) phase is BROADCAST to every registered engine:
         * each fits the same context, and inference can then shard batches
         * across all of them (see pipeline_state_change).  Other models
         * train on a single engine and persist to the shared model repo.
         */
        if (_is_tabpfn(PIPELINE_SESSION.model_name)) {
            int engine_count = 0;
            EngineEndpoint *engines = load_ai_engines(&engine_count);
            int eng_base = 0;
            int eng_used;
            if (engine_count <= 0) {
                free_ai_engines(engines, engine_count);
                elog(ERROR, "nr_aiengine catalog is empty");
            }
            if (nr_engine_pin >= 0) {
                /* session pinned to one engine (task-level parallelism) */
                eng_base = nr_engine_pin % engine_count;
                eng_used = 1;
            } else {
                eng_used = engine_count;
            }
            PIPELINE_SESSION.eng_count = eng_used;
            PIPELINE_SESSION.eng_hosts = (char **) malloc(sizeof(char *) * eng_used);
            PIPELINE_SESSION.eng_ports = (int *) malloc(sizeof(int) * eng_used);
            PIPELINE_SESSION.train_wss =
                (NrWebsocket **) malloc(sizeof(NrWebsocket *) * eng_used);
            if (!PIPELINE_SESSION.eng_hosts || !PIPELINE_SESSION.eng_ports ||
                !PIPELINE_SESSION.train_wss) {
                elog(ERROR, "failed to allocate broadcast-train state");
            }
            for (int i = 0; i < eng_used; i++) {
                EngineEndpoint *e = &engines[eng_base + i];
                PIPELINE_SESSION.eng_hosts[i] = strdup(e->host);
                PIPELINE_SESSION.eng_ports[i] = e->port;
                PIPELINE_SESSION.train_wss[i] =
                    nws_initialize(e->host, e->port, "/ws", 10);
                nws_connect(PIPELINE_SESSION.train_wss[i]);
            }
            free_ai_engines(engines, engine_count);
            PIPELINE_SESSION.ws = PIPELINE_SESSION.train_wss[0];
        } else {
            PIPELINE_SESSION.ws = connect_to_ai_engine();
        }
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
            epoch,
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
        if (_is_tabpfn(PIPELINE_SESSION.model_name)) {
            char *ct = _build_col_types(PIPELINE_SESSION.tupdesc, PIPELINE_SESSION.n_features);
            tt->colTypes = strdup(ct);
            pfree(ct);
        }
        if (PIPELINE_SESSION.train_wss) {
            for (int i = 0; i < PIPELINE_SESSION.eng_count; i++) {
                nws_send_task(PIPELINE_SESSION.train_wss[i], T_TRAIN,
                              PIPELINE_SESSION.table_name, tt);
            }
        } else {
            nws_send_task(PIPELINE_SESSION.ws, T_TRAIN, PIPELINE_SESSION.table_name, tt);
        }
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

        if (PIPELINE_SESSION.train_wss) {
            /*
             * Broadcast-trained in-context model (tabpfn): wait for EVERY
             * engine to finish fitting the context and remember each engine's
             * process-local model id, then shard inference across all of them.
             */
            PIPELINE_SESSION.worker_model_ids =
                (int *) malloc(sizeof(int) * PIPELINE_SESSION.eng_count);
            if (!PIPELINE_SESSION.worker_model_ids) {
                elog(ERROR, "failed to allocate worker model ids");
            }
            for (int i = 0; i < PIPELINE_SESSION.eng_count; i++) {
                nws_wait_completion(PIPELINE_SESSION.train_wss[i]);
                PIPELINE_SESSION.worker_model_ids[i] =
                    PIPELINE_SESSION.train_wss[i]->model_id;
            }
            PIPELINE_SESSION.model_id = PIPELINE_SESSION.worker_model_ids[0];

            /* reset websockets; inference uses fresh per-worker connections */
            for (int i = 0; i < PIPELINE_SESSION.eng_count; i++) {
                _clean_up_conn(PIPELINE_SESSION.train_wss[i]);
            }
            free(PIPELINE_SESSION.train_wss);
            PIPELINE_SESSION.train_wss = NULL;
            PIPELINE_SESSION.ws = NULL;

            PIPELINE_SESSION.dist_infer =
                distributed_infer_create(&PIPELINE_SESSION,
                                         PIPELINE_SESSION.worker_model_ids);
        } else {
            nws_wait_completion(PIPELINE_SESSION.ws);
            PIPELINE_SESSION.model_id = PIPELINE_SESSION.ws->model_id;

            /* reset websocket with a new connection */
            _clean_up_conn(PIPELINE_SESSION.ws);
            PIPELINE_SESSION.ws = NULL;

            PIPELINE_SESSION.dist_infer = distributed_infer_create(&PIPELINE_SESSION, NULL);
        }
        if (!PIPELINE_SESSION.dist_infer) {
            PIPELINE_SESSION.ws = connect_to_ai_engine();
            send_inference_task(PIPELINE_SESSION.ws, &PIPELINE_SESSION, PIPELINE_SESSION.model_id);
        }

        PIPELINE_SESSION.state = PS_INFER;
    } else {
        // INFER -> TRAIN
        // TODO: not supported because ideally there is no INFER -> TRAIN transition
        elog(ERROR, "INFER -> TRAIN state change is not supported");
    }
}


static void
distributed_infer_shutdown(DistributedInfer *dist) {
    if (!dist) {
        return;
    }

    dist->shutting_down = 1;
    close_infer_job_queue(&dist->job_queue);
    close_infer_result_queue(&dist->result_queue);

    pthread_mutex_lock(&dist->result_state.mutex);
    dist->result_state.active = 0;
    pthread_cond_broadcast(&dist->result_state.active_cond);
    pthread_cond_broadcast(&dist->result_state.done_cond);
    pthread_mutex_unlock(&dist->result_state.mutex);

    for (int i = 0; i < dist->worker_count; i++) {
        if (dist->workers[i].thread) {
            pthread_join(dist->workers[i].thread, NULL);
        }
    }
    if (dist->collector_thread) {
        pthread_join(dist->collector_thread, NULL);
    }

    for (int i = 0; i < dist->worker_count; i++) {
        if (dist->workers[i].ws) {
            _clean_up_conn(dist->workers[i].ws);
        }
    }

    destroy_infer_job_queue(&dist->job_queue);
    destroy_infer_result_queue(&dist->result_queue);
    destroy_result_state(&dist->result_state);
    free(dist->workers);
    free(dist);
}

static void
pipeline_close() {
    if (PIPELINE_SESSION.dist_infer) {
        distributed_infer_shutdown(PIPELINE_SESSION.dist_infer);
        PIPELINE_SESSION.dist_infer = NULL;
    }
    if (PIPELINE_SESSION.train_wss) {
        /* error path: query aborted while the broadcast train was in flight */
        for (int i = 0; i < PIPELINE_SESSION.eng_count; i++) {
            if (PIPELINE_SESSION.train_wss[i] == PIPELINE_SESSION.ws) {
                PIPELINE_SESSION.ws = NULL;  /* avoid double free below */
            }
            _clean_up_conn(PIPELINE_SESSION.train_wss[i]);
        }
        free(PIPELINE_SESSION.train_wss);
        PIPELINE_SESSION.train_wss = NULL;
    }
    if (PIPELINE_SESSION.ws) {
        _clean_up_conn(PIPELINE_SESSION.ws);
        PIPELINE_SESSION.ws = NULL;
    }
    if (PIPELINE_SESSION.eng_hosts) {
        for (int i = 0; i < PIPELINE_SESSION.eng_count; i++) {
            free(PIPELINE_SESSION.eng_hosts[i]);
        }
        free(PIPELINE_SESSION.eng_hosts);
        PIPELINE_SESSION.eng_hosts = NULL;
    }
    if (PIPELINE_SESSION.eng_ports) {
        free(PIPELINE_SESSION.eng_ports);
        PIPELINE_SESSION.eng_ports = NULL;
    }
    if (PIPELINE_SESSION.worker_model_ids) {
        free(PIPELINE_SESSION.worker_model_ids);
        PIPELINE_SESSION.worker_model_ids = NULL;
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
    int epoch = PG_GETARG_INT32(3);
    int n_batches = PG_GETARG_INT32(4);
    int nfeat = PG_GETARG_INT32(5);

    ArrayType *arr = PG_GETARG_ARRAYTYPE_P(6);
    int n_features = 0;
    char **feature_names = _array_to_cstring_list(arr, &n_features);

    char *target = text_to_cstring(PG_GETARG_TEXT_P(7));
    PredictType type = PG_GETARG_INT32(8);
    TupleDesc tupdesc = (TupleDesc) PG_GETARG_DATUM(9);

    bool is_inference = pipeline_init(
        model_name,
        table_name,
        batch_size,
        epoch,
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



PG_FUNCTION_INFO_V1(insert_ai_engine);

PG_FUNCTION_INFO_V1(delete_ai_engine);


static Oid
_assign_oid(Relation rel)
{
	return GetNewOidWithIndex(rel,
							  NrAiengineOidIndexId,
							  Anum_nr_aiengine_oid);
}

static HeapTuple
_build_ai_engine_tuple(const char *addr, int port, Relation rel)
{
	Datum		values[Natts_nr_aiengine];
	bool		nulls[Natts_nr_aiengine];

	for (int i = 0; i < Natts_nr_aiengine; i++)
	{
		nulls[i] = true;
	}

	values[Anum_nr_aiengine_oid - 1] = ObjectIdGetDatum(_assign_oid(rel));
	nulls[Anum_nr_aiengine_oid - 1] = false;

	values[Anum_nr_aiengine_aieaddr - 1] = CStringGetTextDatum(addr);
	nulls[Anum_nr_aiengine_aieaddr - 1] = false;

	values[Anum_nr_aiengine_aieport - 1] = Int32GetDatum(port);
	nulls[Anum_nr_aiengine_aieport - 1] = false;

	return heap_form_tuple(RelationGetDescr(rel), values, nulls);
}

Datum
insert_ai_engine(PG_FUNCTION_ARGS)
{
	Relation	rel;
	HeapTuple	tup;

	char	   *addr = text_to_cstring(PG_GETARG_TEXT_P(0));
	int			port = PG_GETARG_INT32(1);

	elog(DEBUG1, "In NeurDB's insert_ai_engine");

	/* Open system catalog nr_aiengine */
	rel = table_open(NrAiengineRelationId, RowExclusiveLock);

	/* add a new tuple into the relation */
	tup = _build_ai_engine_tuple(addr, port, rel);

	Datum		values[2];
	bool		isnull;
	TableScanDesc scan;
	int32		tup_port;
	char        *tup_addr;

	scan = table_beginscan_catalog(rel, 0, NULL);

    bool success = true;

	HeapTuple	curr_tup;
	while ((curr_tup = heap_getnext(scan, ForwardScanDirection)) != NULL)
	{
		values[0] = heap_getattr(curr_tup, Anum_nr_aiengine_aieaddr, rel->rd_att, &isnull);
		if (isnull)
			continue;

		values[1] = heap_getattr(curr_tup, Anum_nr_aiengine_aieport, rel->rd_att, &isnull);
		if (isnull)
			continue;

		tup_addr = text_to_cstring(DatumGetTextPP(values[0]));
		tup_port = DatumGetInt32(values[1]);

		if (tup_port == port &&
			pg_strcasecmp(tup_addr, addr) == 0)
		{
			success = false;
            break;
		}
	}

    if (success)
    {
        CatalogTupleInsert(rel, tup);
        CommandCounterIncrement();
        table_endscan(scan);
        table_close(rel, RowExclusiveLock);
    }
    else
    {
        table_endscan(scan);
        table_close(rel, RowExclusiveLock);
        ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                        errmsg("aiengine %s:%d already exists", addr, port)));
    }

	PG_RETURN_VOID();
}

Datum
delete_ai_engine(PG_FUNCTION_ARGS)
{
	char	   *addr = text_to_cstring(PG_GETARG_TEXT_P(0));
	int			port = PG_GETARG_INT32(1);

	Relation	rel;
	Datum		values[2];
	bool		isnull;
	char		query[1024];

	elog(DEBUG1, "In NeurDB's delete_ai_engine");

	rel = table_open(NrAiengineRelationId, RowExclusiveLock);

	HeapTuple	tup;
	TableScanDesc scan;
	int32		tup_port;
	char        *tup_addr;

	scan = table_beginscan_catalog(rel, 0, NULL);

	bool success = false;

	while ((tup = heap_getnext(scan, ForwardScanDirection)) != NULL)
	{
		values[0] = heap_getattr(tup, Anum_nr_aiengine_aieaddr, rel->rd_att, &isnull);
		if (isnull)
			continue;

		values[1] = heap_getattr(tup, Anum_nr_aiengine_aieport, rel->rd_att, &isnull);
		if (isnull)
			continue;

		tup_addr = text_to_cstring(DatumGetTextPP(values[0]));
		tup_port = DatumGetInt32(values[1]);

		if (tup_port == port &&
			pg_strcasecmp(tup_addr, addr) == 0)
		{
			CatalogTupleDelete(rel, &tup->t_self);
			success = true;
            break;
		}
	}

	table_endscan(scan);

	table_close(rel, RowExclusiveLock);

	if (success)
		PG_RETURN_VOID();
	else
		ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
						errmsg("aiengine %s:%d not found", addr, port)));
}
