#include "action_test.h"
#include "ipc/msg.h"
#include "nram_utils/config.h"
#include "nram_xact/action.h"
#include <sys/time.h>
#include <sys/wait.h>
#include "miscadmin.h"
#include <math.h>

// helper: mirror the encoder logic exactly
static inline uint32
expected_idx(uint32 n_access, uint8 cur_op /* 0=READ,1=WRITE */)
{
    uint32 op  = cur_op ? 1U : 0U;
    uint32 acc = (n_access > 31U) ? 31U : n_access;
    return (op << 5) | acc; /* 0..63 (dep currently ignored) */
}


/*
 * Test basic functionality of NeurCC Agent Function cache
 */
void
run_policy_basic_test(void)
{
    XactFeatureData feat;
    StringInfoData buf;
    CCAction act;
    char tmpfile[] = "/tmp/nram_policy_test.tsv";
    FILE *fp;

    /* Step 1: initialize the shared cache */
    init_agent_function_cache();
    if (GlobalCachedAgentFunc == NULL)
        elog(ERROR, "GlobalCachedAgentFunc not initialized");

    /* Step 2: check defaults at index 0 */
    feat.n_access = 0;
    feat.cur_op   = READ_OPT;
    feat.n_dep    = 0;

    act = get_action(&feat);
    if (act == NULL)
        elog(ERROR, "get_action returned NULL for default feature");

    if (act->detect_all != false || act->priority != 0.0 || act->timeout != DEFAULT_WAIT_NS)
        elog(ERROR, "Default action not set correctly (detect_all=%d, priority=%f, timeout=%u)",
             act->detect_all, act->priority, act->timeout);

    /* Step 3: write a small test file to load custom actions */
    fp = fopen(tmpfile, "w");
    if (!fp)
        elog(ERROR, "Could not open temp file %s", tmpfile);
    fprintf(fp, "1 0 2.71 1000000000\n");
    fprintf(fp, "0 1 3.14 500000000\n"); /* idx detect priority timeout_ns */
    fclose(fp);

    load_agent_function(tmpfile);

    /* Step 4: re-check updated actions */
    feat.n_access = 0;
    feat.cur_op   = READ_OPT;
    act = get_action(&feat);
    if (act->detect_all != true || act->priority != 3.14)
        elog(ERROR, "load_agent_function did not override idx=0 correctly");

    feat.n_access = 1;
    feat.cur_op   = READ_OPT;
    act = get_action(&feat);
    if (act->detect_all != false || act->priority != 2.71)
        elog(ERROR, "load_agent_function did not set idx=1 correctly");

    /* Step 5: print feature string */
    feat.n_access = 5;
    feat.cur_op   = UPDATE_OPT;
    feat.n_dep    = 7;
    initStringInfo(&buf);
    print_xact_feature(&buf, &feat);

    if (strstr(buf.data, "n_access=5") == NULL || strstr(buf.data, "WRITE") == NULL)
        elog(ERROR, "print_xact_feature produced unexpected output: %s", buf.data);

    elog(INFO, "NeurCC agent function test passed!");
}

void
run_policy_full_load_verify_test(void)
{
    init_agent_function_cache();

    char fn[]="/tmp/nram_policy_full.tsv";
    FILE *fp=fopen(fn,"w"); if(!fp) elog(ERROR,"open %s",fn);

    for (uint32 i=0;i<MAX_XACT_FEATURE_SPACE;i++)
    {
        unsigned detect = (i&1U);
        double   prio   = (double)i/7.0;
        unsigned long long tout = (unsigned long long)(i*1234ULL + 7ULL);
        fprintf(fp, "%u %u %.17g %llu\n", i, detect, prio, tout);
    }
    fclose(fp);

    load_agent_function(fn);

    CCActionData *a0 = &GlobalCachedAgentFunc->act[0];
    double exp_prio = 0.0;
    uint32 exp_tout = 7U;

    /* detect_all */
    if (a0->detect_all)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("full verify idx0 failed: detect_all mismatch"),
                errdetail("got=%s expected=false", a0->detect_all ? "true" : "false")));

    /* priority (with epsilon) */
    double diff = fabs(a0->priority - exp_prio);
    if (diff >= 1e-12)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("full verify idx0 failed: priority mismatch"),
                errdetail("got=%.17g expected=%.17g (|diff|=%.3e)", a0->priority, exp_prio, diff)));

    /* timeout */
    if (a0->timeout != exp_tout)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                errmsg("full verify idx0 failed: timeout mismatch"),
                errdetail("got=%u expected=%u", a0->timeout, exp_tout)));

    elog(INFO, "run_policy_full_load_verify_test passed");
}
