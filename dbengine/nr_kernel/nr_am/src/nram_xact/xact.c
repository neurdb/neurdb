#include "nram_xact/xact.h"
#include "utils/memutils.h"
#include "nram_storage/rocksengine.h"
#include "storage/lock.h"

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
    TransactionId xact_id = GetTopTransactionId();
    if (current_nram_xact == NULL || current_nram_xact->xact_id != xact_id)
        current_nram_xact = NewNRAMXactState(xact_id);
}

NRAMXactState GetCurrentNRAMXact(void) {
    refresh_nram_xact();
    return current_nram_xact;
}

static void clear_nram_xact() {
    if (current_nram_xact == NULL) {
        elog(WARNING, "The NRAM transaction has been cleaned before.");
    } else {
        ListCell *cell;
        foreach(cell, current_nram_xact->read_set) {
            NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
            pfree(opt);
        }
        list_free(current_nram_xact->read_set);
        foreach(cell, current_nram_xact->write_set) {
            NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
            pfree(opt);
        }
        list_free(current_nram_xact->write_set);
        pfree(current_nram_xact);
        current_nram_xact = NULL;
    }
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
                    current_nram_xact->xact_id);
            } else {
                // LOCKTAG		tag;
                KVEngine* engine = GetCurrentEngine();
                ListCell *cell;
                // foreach(cell, current_nram_xact->write_set) {
                //     NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);                    
	            //     // SET_LOCKTAG_NRAM_OPT(tag, opt->key);
                //     LockAcquire(&tag, ExclusiveLock, true, false);
                // }

                foreach(cell, current_nram_xact->read_set) {
                    NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
                    NRAMValue cur_val = rocksengine_get(engine, opt->key);

                    if (cur_val == NULL) {
                        elog(ERROR, "transaction validation failed: key vanished");
                        return;
                    }
                    if (cur_val->xact_id != opt->xact_id)
                        elog(ERROR,
                            "The transaction %u gets aborted during read set validation.",
                            current_nram_xact->xact_id);
                }

                NRAM_TEST_INFO("The validation has been passed");
                current_nram_xact->validated = true;
            }
            // Add pre-commit validation or flush logic here
            break;

        case XACT_EVENT_ABORT:
            NRAM_TEST_INFO("the transaction %u is aborted",
                           current_nram_xact->xact_id);
            clear_nram_xact();
            break;

        case XACT_EVENT_COMMIT:
            NRAM_TEST_INFO("the transaction %u is committed",
                           current_nram_xact->xact_id);
            clear_nram_xact();
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


NRAMXactState NewNRAMXactState(TransactionId xact_id) {
    MemoryContext oldCtx;
    NRAMXactState res = NULL;
    if (xact_id == InvalidTransactionId)
        return res;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    res = palloc(sizeof(NRAMXactStateData));
    
    res->xact_id = xact_id;
    // res->begin_ts = GetCurrentTransactionStartTimestamp();
    res->validated = false;
    res->read_set = NIL;
    res->write_set = NIL;

    MemoryContextSwitchTo(oldCtx);
    return res;
}


void add_read_set(NRAMXactState state, NRAMKey key, TransactionId xact_id) {
    NRAMXactOpt opt = palloc(sizeof(NRAMXactOptData));
    NRAM_INFO();
    opt->key = key;
    opt->xact_id = xact_id;
    opt->type = XACT_OP_READ;
    opt->value = NULL;
    state->read_set = lappend(state->read_set, opt);
}

void add_write_set(NRAMXactState state, NRAMKey key, NRAMValue value) {
    NRAMXactOpt opt = palloc(sizeof(NRAMXactOptData));
    opt->key = key;
    opt->xact_id = state->xact_id;
    opt->value = value;
    opt->type = XACT_OP_WRITE;

    state->write_set = lappend(state->write_set, opt);
}

bool validate_read_set(KVEngine* engine, NRAMXactState state) {
    ListCell *cell;

    foreach(cell, state->read_set) {
        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
        NRAMValue cur_val = rocksengine_get(engine, opt->key);

        if (cur_val == NULL) {
            NRAM_TEST_INFO("validation failed: key vanished");
            return false;
        }

        if (cur_val->xact_id != opt->xact_id)
            return false;
    }
    return true;
}