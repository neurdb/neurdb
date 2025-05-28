#include "nram_xact/xact.h"
#include "utils/memutils.h"

/* ------------------------------------------------------------------------
 * Transaction related codes.
 * ------------------------------------------------------------------------
 */

static bool xact_hook_registered = false;
static NRAMXactState current_nram_xact = NULL;

void refresh_nram_xact(void) {
    TransactionId tid = GetTopTransactionIdIfAny();
    if (current_nram_xact == NULL || current_nram_xact->tid != tid)
        current_nram_xact = NewNRAMXactState(tid);
}

static void nram_xact_callback(XactEvent event, void *arg) {
    MemoryContext oldCtx;
    if (current_nram_xact == NULL)
        return;

    oldCtx = MemoryContextSwitchTo(TopTransactionContext);
    NRAM_TEST_INFO("The callback is on %d", event);
    refresh_nram_xact();

    switch (event) {
        case XACT_EVENT_PRE_COMMIT:
            if (current_nram_xact->validated) {
                NRAM_TEST_INFO(
                    "the transaction %u has already been validated before",
                    current_nram_xact->tid);
            }
            elog(INFO, "[nram] Pre-commit hook triggered");
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
    res->read_set = NULL;
    res->write_set = NULL;

    MemoryContextSwitchTo(oldCtx);
    return res;
}
