#include "nram_xact/action.h"
#include "utils/memutils.h"
#include "storage/shmem.h"
#include "miscadmin.h"
#include "lib/stringinfo.h"
#include "access/xact.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

CachedAgentFunc GlobalCachedAgentFunc = NULL;

/* ---------- Local helpers ---------- */

static inline uint32
min_u32(uint32 x, uint32 y)
{
    return (x < y) ? x : y;
}

/* 10-bit encoding:
 * [9]      : cur_op (1 bit)
 * [8..4]   : n_access (5 bits, 0..31, clamped)
 * --> [3..0]   : n_dep (2 bits, 0..15, clamped)
 * Total    : 10 bits -> 0..1023 == MAX_XACT_FEATURE_SPACE-1
 */
static inline uint32
encode_feature_10bit(const XactFeature feature)
{
    uint32 op  = (feature->cur_op ? 1U : 0U);              /* 1 bit  */
    uint32 acc = min_u32(feature->n_access, 31U);          /* 5 bits */
    // uint32 dep = min_u32(feature->n_dep,    15U);          /* 4 bits */
    // disabled by now.

    uint32 idx = 0;
    idx |= op;
    idx <<= 5;
    idx |= acc;

    return idx; /* 0..64 */
}

/* ---------- Shared state backing ---------- */

/* Name for ShmemInitStruct so multiple backends attach to the same block */
#define NRAM_AGENTFUNC_SHMEM_NAME "NeurCC_AgentFuncCache"

/* ---------- Public API impls (per .h) ---------- */

void
init_agent_function_cache(void)
{
    bool    found = false;

    /* Allocate/attach a single shared CachedAgentFuncData block */
    CachedAgentFuncData *ptr = (CachedAgentFuncData *)
        ShmemInitStruct(NRAM_AGENTFUNC_SHMEM_NAME,
                        sizeof(CachedAgentFuncData),
                        &found);

    if (!found)
    {
        /* First initializer: zero-init and set conservative defaults */
        MemSet(ptr, 0, sizeof(CachedAgentFuncData));
        for (uint32 i = 0; i < MAX_XACT_FEATURE_SPACE; ++i)
        {
            ptr->act[i].detect_all = false;
            ptr->act[i].priority   = 0.0;   /* neutral priority */
            ptr->act[i].timeout    = DEFAULT_WAIT_NS;     /* 1s wait by default */
        }
    }

    /* Publish the pointer set in the header (declared there, defined there) */
    GlobalCachedAgentFunc = ptr;
}

void
load_agent_function(const char *filename)
{
    FILE *fp;
    char  line[256];
    uint32 lineno = 0;

    if (GlobalCachedAgentFunc == NULL)
        ereport(ERROR,
                (errmsg("GlobalCachedAgentFunc not initialized"),
                 errhint("Call init_agent_function_cache() during shared memory startup.")));

    if (filename == NULL || filename[0] == '\0')
        ereport(ERROR,
                (errmsg("load_agent_function: invalid filename")));

    fp = fopen(filename, "r");
    if (fp == NULL)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("load_agent_function: could not open \"%s\": %m", filename)));

    /* Simple, robust TSV/space-separated format per line:
     * index  detect_all  priority  timeout_ns
     * - index: 0..1023 (optional; if omitted, we compute from a running cursor)
     * - detect_all: 0/1
     * - priority: double
     * - timeout_ns: uint64 (nanoseconds)
     *
     * If the file instead provides triplets (detect_all, priority, timeout_ns)
     * without index, we assign them sequentially from 0 upward.
     */
    uint32 cursor = 0;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        lineno++;

        /* Skip comments and blank lines */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '#')
            continue;

        /* Try reading 4 fields (with explicit index) */
        uint32 idx, detect;
        double prio;
        unsigned long long tout;

        int n = sscanf(p, "%u %u %lf %llu", &idx, &detect, &prio, &tout);
        if (n == 4)
        {
            if (idx >= MAX_XACT_FEATURE_SPACE)
                ereport(ERROR,
                        (errmsg("load_agent_function: index %u out of range at line %u", idx, lineno)));

            GlobalCachedAgentFunc->act[idx].detect_all = (detect != 0);
            GlobalCachedAgentFunc->act[idx].priority   = prio;
            GlobalCachedAgentFunc->act[idx].timeout    = (uint32) Min(tout, (unsigned long long) UINT32_MAX);
            continue;
        }

        /* Try reading 3 fields (no index, use cursor) */
        n = sscanf(p, "%u %lf %llu", &detect, &prio, &tout);
        if (n == 3)
        {
            if (cursor >= MAX_XACT_FEATURE_SPACE)
                ereport(ERROR,
                        (errmsg("load_agent_function: too many entries (>%d)", MAX_XACT_FEATURE_SPACE)));

            GlobalCachedAgentFunc->act[cursor].detect_all = (detect != 0);
            GlobalCachedAgentFunc->act[cursor].priority   = prio;
            GlobalCachedAgentFunc->act[cursor].timeout    = (uint32) Min(tout, (unsigned long long) UINT32_MAX);
            cursor++;
            continue;
        }

        ereport(ERROR,
                (errmsg("load_agent_function: malformed line %u in \"%s\"", lineno, filename),
                 errdetail("Expected: 'idx detect_all priority timeout_ns' or 'detect_all priority timeout_ns'.")));
    }

    fclose(fp);
}

CCAction
get_action(XactFeature feature)
{
    if (GlobalCachedAgentFunc == NULL)
        ereport(ERROR,
                (errmsg("GlobalCachedAgentFunc not initialized"),
                 errhint("Call init_agent_function_cache() during shared memory startup.")));

    if (feature == NULL)
        ereport(ERROR, (errmsg("get_action: feature is NULL")));

    uint32 idx = encode_feature_10bit(feature);
    return &GlobalCachedAgentFunc->act[idx];
}

void
print_xact_feature(StringInfo str, XactFeature feature)
{
    if (feature == NULL)
    {
        appendStringInfoString(str, "<XactFeature NULL>");
        return;
    }

    appendStringInfo(
        str,
        "{n_access=%u, cur_op=%s, n_dep=%u, enc_idx=%u}",
        feature->n_access,
        (feature->cur_op == READ_OPT) ? "READ" :
        (feature->cur_op == UPDATE_OPT) ? "WRITE" : "UNKNOWN",
        feature->n_dep,
        encode_feature_10bit(feature)
    );
}

/* ---------- Shared memory sizing helper (optional) ----------
 * If you hook into Postgres' shmem request path, call this to reserve bytes.
 */
Size
NeurCC_AgentFuncCacheShmemSize(void)
{
    return add_size(sizeof(CachedAgentFuncData), 0);
}
