#include "nram_xact/xact.h"
#include "utils/memutils.h"
#include "nram_storage/rocksengine.h"

/* ------------------------------------------------------------------------
 * Transaction related codes.
 * ------------------------------------------------------------------------
 */

static bool xact_hook_registered = false;
static NRAMXactState current_nram_xact = NULL;

const char* XactEventString[] = {
	"COMMIT",
	"PARALLEL_COMMIT",
	"ABORT",
	"PARALLEL_ABORT",
	"PREPARE",
	"PRE_COMMIT",
	"PARALLEL_PRE_COMMIT",
	"PRE_PREPARE"
};

void refresh_nram_xact(void) {
    TransactionId tid = GetTopTransactionId();
    if (current_nram_xact == NULL || current_nram_xact->tid != tid)
        current_nram_xact = NewNRAMXactState(tid);
}

NRAMXactState GetCurrentNRAMXact(void) {
    refresh_nram_xact();
    return current_nram_xact;
}

static void nram_xact_callback(XactEvent event, void *arg) {
    MemoryContext oldCtx;
    if (current_nram_xact == NULL)
        return;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    NRAM_TEST_INFO("The callback is triggered on event %s", XactEventString[event]);
    refresh_nram_xact();

    switch (event) {
        case XACT_EVENT_PRE_COMMIT:
            if (current_nram_xact->validated) {
                elog(ERROR,
                    "The transaction %u has already been validated before.",
                    current_nram_xact->tid);
            } else {
                if (!validate_read_set(GetCurrentEngine(), GetCurrentNRAMXact()))
                    elog(ERROR,
                        "The transaction %u gets aborted during read set validation.",
                        current_nram_xact->tid);
                NRAM_TEST_INFO("The validation has been passed");
                current_nram_xact->validated = true;
            }
            // Add pre-commit validation or flush logic here
            break;

        case XACT_EVENT_ABORT:
            NRAM_TEST_INFO("the transaction %u is aborted",
                           current_nram_xact->tid);
            break;

        case XACT_EVENT_COMMIT:
            NRAM_TEST_INFO("the transaction %u is committed",
                           current_nram_xact->tid);
            break;

        default:
            break;
    }

    MemoryContextSwitchTo(oldCtx);
}

void nram_register_xact_hook(void) {
    if (!xact_hook_registered) {
        RegisterXactCallback(nram_xact_callback, NULL);
        xact_hook_registered = true;
    }
}

void nram_unregister_xact_hook(void) {
    if (!xact_hook_registered) {
        UnregisterXactCallback(nram_xact_callback, NULL);
        xact_hook_registered = false;
    }
}


NRAMXactState NewNRAMXactState(TransactionId tid) {
    MemoryContext oldCtx;
    NRAMXactState res = NULL;
    if (tid == InvalidTransactionId)
        return res;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    res = palloc(sizeof(NRAMXactStateData));
    
    res->tid = tid;
    // res->begin_ts = GetCurrentTransactionStartTimestamp();
    res->validated = false;
    res->read_set = NIL;
    res->write_set = NIL;

    MemoryContextSwitchTo(oldCtx);
    return res;
}


void add_read_set(NRAMXactState state, NRAMKey key, TransactionId tid) {
    NRAMXactOpt opt = palloc(sizeof(NRAMXactOptData));
    opt->key = key;
    opt->tid = tid;
    opt->type = XACT_OP_READ;
    opt->value = NULL;
    state->read_set = lappend(state->read_set, opt);
}

void add_write_set(NRAMXactState state, NRAMKey key, NRAMValue value) {
    NRAMXactOpt opt = palloc(sizeof(NRAMXactOptData));
    opt->key = key;
    opt->tid = state->tid;
    opt->value = value;
    opt->type = XACT_OP_WRITE;

    state->write_set = lappend(state->write_set, opt);
}

bool validate_read_set(KVEngine* engine, NRAMXactState state) {
    ListCell *cell;
    NRAM_INFO();
    foreach(cell, state->read_set) {
        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
        NRAMValue cur_val = rocksengine_get(engine, opt->key);

        NRAM_INFO();
        if (cur_val == NULL) {
            NRAM_TEST_INFO("validation failed: key vanished");
            return false;
        }

        if (cur_val->tid != opt->tid)
            return false;
    }
    return true;
}