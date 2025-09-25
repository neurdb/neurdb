#include "action_test.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"
#include "nram_xact/action.h"
#include <sys/time.h>
#include <sys/wait.h>
#include "miscadmin.h"
#include <math.h>

// helper: mirror the encoder logic exactly
static inline uint32 expected_idx(uint32 n_access,
                                  uint8 cur_op /* 0=READ,1=WRITE */) {
    uint32 op = cur_op ? 1U : 0U;
    uint32 acc = (n_access > 31U) ? 31U : n_access;
    return (op << 5) | acc; /* 0..63 (dep currently ignored) */
}

/*
 * Test basic functionality of NeurCC Agent Function cache
 */
void run_policy_basic_test(void) {
    XactFeatureData feat;
    StringInfoData buf;
    CCAction act;
    char tmpfile[] = "/tmp/nram_policy_test.tsv";
    FILE *fp;

    /* Step 1: initialize the shared cache upon startup */
    if (GlobalCachedAgentFunc == NULL)
        elog(ERROR, "GlobalCachedAgentFunc not initialized");

    /* Step 2: check defaults at index 0 */
    feat.n_access = 0;
    feat.cur_op = READ_OPT;
    feat.n_dep = 0;

    act = get_action(&feat);
    if (act == NULL)
        elog(ERROR, "get_action returned NULL for default feature");

    if (act->detect_all != false || act->priority != 0.0 ||
        act->timeout != DEFAULT_WAIT_NS)
        elog(ERROR,
             "Default action not set correctly (detect_all=%d, priority=%f, "
             "timeout=%u)",
             act->detect_all, act->priority, act->timeout);

    /* Step 3: write a small test file to load custom actions */
    fp = fopen(tmpfile, "w");
    if (!fp) elog(ERROR, "Could not open temp file %s", tmpfile);
    fprintf(fp, "1 0 2.71 1000000000\n");
    fprintf(fp, "0 1 3.14 500000000\n"); /* idx detect priority timeout_ns */
    fclose(fp);

    load_agent_function(tmpfile);

    /* Step 4: re-check updated actions */
    feat.n_access = 0;
    feat.cur_op = READ_OPT;
    act = get_action(&feat);
    if (act->detect_all != true || act->priority != 3.14)
        elog(ERROR, "load_agent_function did not override idx=0 correctly");

    feat.n_access = 1;
    feat.cur_op = READ_OPT;
    act = get_action(&feat);
    if (act->detect_all != false || act->priority != 2.71)
        elog(ERROR, "load_agent_function did not set idx=1 correctly");

    /* Step 5: print feature string */
    feat.n_access = 5;
    feat.cur_op = UPDATE_OPT;
    feat.n_dep = 7;
    initStringInfo(&buf);
    print_xact_feature(&buf, &feat);

    if (strstr(buf.data, "n_access=5") == NULL ||
        strstr(buf.data, "WRITE") == NULL)
        elog(ERROR, "print_xact_feature produced unexpected output: %s",
             buf.data);

    elog(INFO, "Agent function basic test passed!");
}

void
run_policy_full_load_verify_test(void)
{
    const char *fn = "/tmp/nram_policy_full.tsv";
    FILE *fp;

    /* Make sure shared cache is ready (ok in tests; in prod use shmem_startup_hook) */
   if (GlobalCachedAgentFunc == NULL)
        elog(ERROR, "GlobalCachedAgentFunc not initialized");

    fp = fopen(fn, "w");
    if (!fp)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("failed to open %s for write", fn)));

    for (uint32 i = 0; i < MAX_XACT_FEATURE_SPACE; i++)
    {
        uint32 detect   = (i & 1U);
        double prio     = (double)i / 7.0;
        uint64 tout64   = (uint64)i * 1234U + 7U;  /* portable 64-bit */

        /* If your loader expects uint32 timeout, clamp here to avoid wrap surprises */
        /* uint32 tout32 = (uint32) tout64; */

        /* index detect priority timeout */
        fprintf(fp, "%u %u %.17g " UINT64_FORMAT "\n", i, detect, prio, tout64);
    }

    if (fclose(fp) != 0)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("failed to close %s after write", fn)));

    /* Load into the shared cache */
    load_agent_function(fn);

    /* Verify slots */
    for (uint32 i = 0; i < MAX_XACT_FEATURE_SPACE; i++)
    {
        XactFeatureData feat;
        CCActionData *act;
        const double exp_prio = (double)i / 7.0;
        const uint32 exp_tout = (uint32)i * 1234U + 7U;  /* loader may cast to uint32 */
        const bool exp_detect = (i & 1U) ? true : false;
        uint32 idx;

        feat.n_access = i & 31U;  /* max 31 */
        feat.cur_op = (i & 2U) ? UPDATE_OPT : READ_OPT;
        feat.n_dep = (i >> 2) & 15U;  /* max 15, currently unused */

        idx = expected_idx(feat.n_access, feat.cur_op);
        if (idx != i)
            continue;  /* skip unused slots */

        act = &GlobalCachedAgentFunc->act[idx];
        if (act == NULL)
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("get_action returned NULL for feature index %u", i)));

        /* detect_all */
        if (act->detect_all != exp_detect)
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("full verify idx%u failed: detect_all mismatch", i),
                     errdetail("got=%s expected=%s",
                               act->detect_all ? "true" : "false",
                               exp_detect ? "true" : "false")));

        /* priority with epsilon */
        {
            double diff = fabs(act->priority - exp_prio);
            if (diff >= 1e-12)
                ereport(ERROR,
                        (errcode(ERRCODE_INTERNAL_ERROR),
                         errmsg("full verify idx%u failed: priority mismatch", i),
                         errdetail("got=%.17g expected=%.17g (|diff|=%.3e)",
                                   act->priority, exp_prio, diff)));
        }

        /* timeout: loader may cast to uint32 */
        if (act->timeout != exp_tout)
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("full verify idx%u failed: timeout mismatch", i),
                     errdetail("got=%u expected=%u", act->timeout, exp_tout)));
    }

    elog(INFO, "Agent function verification test passed");
}