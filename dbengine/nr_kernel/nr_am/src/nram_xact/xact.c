#include "nram_xact/xact.h"
#include "utils/memutils.h"
#include "nram_storage/rocks_handler.h"
#include "storage/lock.h"
#include "storage/proc.h"

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

/* ------------------------------------------------------------------------
 * Transaction states & operations helper functions.
 * ------------------------------------------------------------------------
 */

 NRAMXactState NewNRAMXactState(TransactionId xact_id) {
    MemoryContext oldCtx;
    NRAMXactState res = NULL;
    if (xact_id == InvalidTransactionId)
        return res;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    res = palloc0(sizeof(NRAMXactStateData));

    res->xact_id = xact_id;
    res->validated = false;
    res->read_set = NIL;
    res->write_set = NIL;

    MemoryContextSwitchTo(oldCtx);
    return res;
}

void refresh_nram_xact(void) {
    TransactionId xact_id = GetTopTransactionId();
    if (current_nram_xact == NULL || current_nram_xact->xact_id != xact_id)
        current_nram_xact = NewNRAMXactState(xact_id);
}

void nram_lock(NRAMKey key, LOCKMODE mode) {
    LOCKTAG tag;
    LockAcquireResult res;
    SET_LOCKTAG_ADVISORY(tag, key->tableOid, key->tid, 0, 0);
    res = LockAcquire(&tag, mode, false, false);
    if (res == LOCKACQUIRE_NOT_AVAIL) {
        elog(ERROR, "Failed to acquire advisory lock for <%d:%lu>",
             key->tableOid, key->tid);
    }
}

bool nram_release(NRAMKey key, LOCKMODE mode) {
    LOCKTAG tag;
    SET_LOCKTAG_ADVISORY(tag, key->tableOid, key->tid, 0, 0);
    return LockRelease(&tag, mode, false);
}


LockAcquireResult nram_try_lock(NRAMKey key, LOCKMODE mode) {
    LOCKTAG tag;
    SET_LOCKTAG_ADVISORY(tag, key->tableOid, key->tid, 0, 0);
    return LockAcquire(&tag, mode, false, true);
}

NRAMXactState GetCurrentNRAMXact(void) {
    refresh_nram_xact();
    return current_nram_xact;
}

static inline void clear_nram_xact() {
    if (current_nram_xact == NULL) {
        elog(WARNING, "The NRAM transaction has been cleaned before.");
    } else {
        ListCell *cell;
        foreach(cell, current_nram_xact->read_set) {
            NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
            pfree(opt->key);
            pfree(opt->value);
            pfree(opt);
        }
        list_free(current_nram_xact->read_set);
        foreach(cell, current_nram_xact->write_set) {
            NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
            pfree(opt->key);
            pfree(opt->value);
            pfree(opt);
        }
        list_free(current_nram_xact->write_set);
        pfree(current_nram_xact);
        current_nram_xact = NULL;
    }
}

static int nram_opt_cmp(const ListCell *a, const ListCell *b) {
    const NRAMXactOpt opt1 = (NRAMXactOpt) lfirst(a);
    const NRAMXactOpt opt2 = (NRAMXactOpt) lfirst(b);
    if (opt1->key->tableOid != opt2->key->tableOid)
        return (opt1->key->tableOid < opt2->key->tableOid) ? -1 : 1;
    if (opt1->key->tid < opt2->key->tid) return -1;
    if (opt1->key->tid > opt2->key->tid) return 1;
    return 0;
}

static inline bool nram_key_equal(NRAMKey k1, NRAMKey k2) {
    return k1->tableOid == k2->tableOid && k1->tid == k2->tid;
}



/* ------------------------------------------------------------------------
 * Transaction read/write set maintenance.
 * ------------------------------------------------------------------------
 */



NRAMXactOpt find_write_set(NRAMXactState state, NRAMKey key) {
    ListCell *cell;
    NRAM_TEST_INFO("write_set=%p length=%d",
        state->write_set, list_length(state->write_set));

    foreach(cell, state->write_set) {
        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
        if (nram_key_equal(opt->key, key))
            return opt;
    }
    return NULL;
}

NRAMXactOpt find_read_set(NRAMXactState state, NRAMKey key) {
    ListCell *cell;
    foreach(cell, state->read_set) {
        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
        if (nram_key_equal(opt->key, key))
            return opt;
    }
    return NULL;
}

void add_read_set(NRAMXactState state, NRAMKey key, NRAMValue value) {
    MemoryContext oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    NRAMXactOpt opt;

    Assert(find_read_set(state, key) == NULL);
    Assert(find_write_set(state, key) == NULL);
    opt = palloc(sizeof(NRAMXactOptData));
    opt->key = copy_nram_key(key);
    opt->xact_id = value->xact_id;
    opt->value = copy_nram_value(value);
    opt->type = XACT_OP_READ;
    state->read_set = lappend(state->read_set, opt);
    MemoryContextSwitchTo(oldCtx);
}

void add_write_set(NRAMXactState state, NRAMKey key, NRAMValue value) {
    NRAMXactOpt opt;
    MemoryContext oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    NRAMXactOpt find_opt = find_write_set(state, key);

    if (find_opt == NULL) {
        opt = palloc(sizeof(NRAMXactOptData));
        opt->key = copy_nram_key(key);
        opt->xact_id = state->xact_id;
        opt->value = copy_nram_value(value);
        opt->type = XACT_OP_WRITE;
        state->write_set = lappend(state->write_set, opt);
        Assert(find_write_set(state, key) == opt);
    } else {
        pfree(find_opt->value);
        find_opt->value = copy_nram_value(value);
    }
    MemoryContextSwitchTo(oldCtx);
}

bool read_own_write(NRAMXactState state, const NRAMKey key, NRAMValue *value) {
    NRAMXactOpt tmp = find_write_set(state, key);
    if (tmp == NULL)
        return false;
    *value = copy_nram_value(tmp->value);
    return true;
}

bool read_own_read(NRAMXactState state, const NRAMKey key, NRAMValue *value) {
    NRAMXactOpt tmp = find_read_set(state, key);
    if (tmp == NULL)
        return false;
    *value = copy_nram_value(tmp->value);
    return true;
}

static void nram_xact_callback(XactEvent event, void *arg) {
    MemoryContext oldCtx;
    if (current_nram_xact == NULL)
        return;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    NRAM_TEST_INFO("The callback is triggered on event %s", XactEventString[event]);
    refresh_nram_xact();

    switch (event) {
        case XACT_EVENT_PRE_COMMIT: {
            if (current_nram_xact->validated) {
                elog(ERROR,
                    "The transaction %u has already been validated before.",
                    current_nram_xact->xact_id);
            } else {
                ListCell *cell;

                list_sort(current_nram_xact->write_set, nram_opt_cmp);
                NRAM_TEST_INFO("Pre-validation add write locks.");
                foreach(cell, current_nram_xact->write_set) {
                    NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
                    nram_lock(opt->key, ExclusiveLock);
                }

                if (IsolationIsSerializable()) {
                    // Validate the read operations for serializable isolation level.
                    NRAM_TEST_INFO("The validation is processing, validating read values.");
                    foreach(cell, current_nram_xact->read_set) {
                        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
                        LockAcquireResult r;

                        // During read, we abort for pending writes. (peek lock here).
                        r = nram_try_lock(opt->key, ShareLock);
                        if (r == LOCKACQUIRE_NOT_AVAIL) {
                            elog(ERROR,
                                "The transaction %u validation failed: concurrent update detected",
                                current_nram_xact->xact_id);
                        }

                        NRAMValue cur_val = RocksClientGet(opt->key);
                        if (cur_val == NULL) {
                            elog(ERROR,
                                "The transaction %u validation failed: invisible key",
                                current_nram_xact->xact_id);
                            return;
                        }

                        if (cur_val->xact_id != opt->xact_id) {
                            ereport(ERROR,
                                    (errmsg("Transaction aborted during read set validation."),
                                    errdetail("Transaction ID: %u", current_nram_xact->xact_id)));
                        }

                        // We acquired the lock successfully & we are not already the lock owner, meaning no one is writing to it.
                        // We can release the lock immediately.
                        if (r == LOCKACQUIRE_OK)
                            nram_release(opt->key, ShareLock);
                    }
                }

                NRAM_TEST_INFO("Post-validation release write locks.");
                current_nram_xact->validated = true;
                foreach(cell, current_nram_xact->write_set) {
                    NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
                    RocksClientPut(opt->key, opt->value);   // deferred write for OCC.
                    nram_release(opt->key, ExclusiveLock);
                }
                // TODO: flush to WAL here.
                NRAM_TEST_INFO("Validation succeed.");
            }
            break;
        }

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

bool validate_read_set(NRAMXactState state) {
    ListCell *cell;

    foreach(cell, state->read_set) {
        NRAMXactOpt opt = (NRAMXactOpt) lfirst(cell);
        NRAMValue cur_val = RocksClientGet(opt->key);

        if (cur_val == NULL) {
            NRAM_TEST_INFO("validation failed: key vanished");
            return false;
        }

        if (cur_val->xact_id != opt->xact_id)
            return false;
    }
    return true;
}


// To support NeurCC, this function is called before each access (read or write).
void before_access(NRAMXactState state, bool is_write) {
    NRAM_INFO();
    if (state->feature == NULL) {
        MemoryContext oldCtx = MemoryContextSwitchTo(TopTransactionContext);
        state->feature = palloc0(sizeof(XactFeatureData));
        state->feature->n_access = 0;
        state->feature->cur_op = 0;
        state->feature->n_dep = MyProc->nDep;
        MemoryContextSwitchTo(oldCtx);
    } else {
        state->feature->n_access += 1;
        if (is_write)
            state->feature->cur_op = UPDATE_OPT;
        state->feature->n_dep = MyProc->nDep;
    }
    state->action = get_action(state->feature);
    MyProc->rank = state->action->priority;    // set the wait priority.
    LockTimeout = state->action->timeout;  // set the wait timeout.
}

