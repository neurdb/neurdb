#include "postgres.h"
#include <string.h>
#include "fmgr.h"
#include "utils/geo_decls.h"
#include "bloom.h"
#include "funcapi.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

/* Some hyperparameters */
int DYNAMIC = 2;
int DYNMIAC_DETECT_FREQ = 1000;
float MIN_ACTIVATE_PRUNE_RATE = 0.1;

/* Define global variables */
struct bloom* bl_ptrs[MAX_BLOOM_FILTERS] = { NULL };
int bloom_disabled_flag[MAX_BLOOM_FILTERS] = { -1 };
int bloom_cnters[MAX_BLOOM_FILTERS];
float bloom_prune_rates[MAX_BLOOM_FILTERS] = { -1 };
int n_bloom_used = 0;
int shared_mem_location = 0;

/* For postgres */
PG_MODULE_MAGIC;

/*******************************************************************
* Function: _PG_init()
* Description: the standard PG extension initilization
* Parameters: void
* Return: void
*******************************************************************/
void _PG_init(void)
{
    return;
}

/*******************************************************************
* Function: pg_lip_bloom_make_shared()
* Description: move the bloom filters from local memory to shared memory
* Parameters: void
* Return: void
*******************************************************************/
static void pg_lip_bloom_make_shared(void)
{
    int i;
    struct bloom * new_shared_blm;
    struct bloom * local_blm;
    unsigned char * new_shared_blm_bf_data;
    for (i = 0; i < n_bloom_used; i++){
        local_blm = bl_ptrs[i];
        new_shared_blm = create_shared_bloom_ptr(i);
        memcpy(new_shared_blm, local_blm, sizeof(struct bloom));

        new_shared_blm_bf_data = create_bf_data_space(i, local_blm->bytes);
        memcpy(new_shared_blm_bf_data, local_blm->bf, local_blm->bytes);
        shmdt(new_shared_blm_bf_data);
        shmdt(new_shared_blm);
        bloom_free(local_blm);
        free(local_blm);
        bl_ptrs[i] = NULL;
    }
}

/*******************************************************************
* Function: pg_lip_bloom_make_local()
* Description: fetch the memory pointer from the shared buffer
* Parameters: int idx - the index of the bloom filter
* Return: void (the fetched pointer is stored in the local pointer array)
*******************************************************************/
static void pg_lip_bloom_make_local(int idx){
    struct bloom * new_bloom = (struct bloom*) malloc(sizeof(struct bloom));
    struct bloom * shared_bloom = get_shared_bloom_ptr(idx);
    unsigned char * shared_bf_data = get_bf_data_space(idx, shared_bloom->bytes);
    new_bloom = bloom_cpy(
            new_bloom,
            NULL,
            shared_bloom,
            shared_bf_data);
    shmdt(shared_bloom);
    bl_ptrs[idx] = new_bloom;
    bl_ptrs[idx] -> bf = shared_bf_data;
    bl_ptrs[idx] -> probe_cnt = 0;
    bl_ptrs[idx] -> prune_cnt = 0;
    bl_ptrs[idx] -> total_probe_cnt = 0;
    if (DYNAMIC != 0){
        bloom_disabled_flag[idx] = -1;
    }
    else{
        bloom_disabled_flag[idx] = 1;
    }
}

/*******************************************************************
* Function: pg_lip_bloom_set_dynamic()
* Description: set the dynamic state.
*       0 for static probing,
        1 for dynmic probing and stats are collected from the first DYNMIAC_DETECT_FREQ probes
        2 for dynamic probing and stats are collected every 10 * DYNMIAC_DETECT_FREQ probes
* Parameters:
        int set_dynmiac - the value of the dynamic states
* Return:
        void
*******************************************************************/
PG_FUNCTION_INFO_V1(pg_lip_bloom_set_dynamic);
Datum
pg_lip_bloom_set_dynamic(PG_FUNCTION_ARGS)
{
    int set_dynmiac = PG_GETARG_INT32(0);
    if (set_dynmiac < 0 || set_dynmiac > 2)
        ereport(ERROR,
                (errmsg("dynamic mode must be 0, 1, or 2")));
    DYNAMIC = set_dynmiac;
    PG_RETURN_INT32(0);
}

/*******************************************************************
* Function: pg_lip_bloom_init()
* Description: initialize bloom filters
* Parameters:
        int n_bloom - the number of bloom filters needed
* Return:
        void
*******************************************************************/
PG_FUNCTION_INFO_V1(pg_lip_bloom_init);
Datum
pg_lip_bloom_init(PG_FUNCTION_ARGS)
{
    int n_bloom = PG_GETARG_INT32(0);
    int i;
    struct bloom * new_bloom;

    if (n_bloom <= 0){
        n_bloom_used = MAX_BLOOM_FILTERS;
    } else if (n_bloom <= MAX_BLOOM_FILTERS) {
        n_bloom_used = n_bloom;
    } else {
        ereport(ERROR,
                (errmsg("requested %d Bloom filters; maximum is %d",
                        n_bloom, MAX_BLOOM_FILTERS)));
    }

    for (i = 0; i < MAX_BLOOM_FILTERS; i++){
        if (bl_ptrs[i] != NULL){
            if (bl_ptrs[i]->bf != NULL)
                shmdt(bl_ptrs[i]->bf);
            free(bl_ptrs[i]);
            bl_ptrs[i] = NULL;
        }
    }

    for (i = 0; i < MAX_BLOOM_FILTERS; i++){
        if (i < n_bloom_used ){
            new_bloom = (struct bloom*) malloc(sizeof(struct bloom));
            if (new_bloom == NULL ||
                bloom_init(new_bloom, MAX_BLOOM_SIZE, 0.01, i) != 0)
                ereport(ERROR,
                        (errmsg("could not initialize Bloom filter %d", i)));
            bl_ptrs[i] = new_bloom;
            bloom_cnters[i] = 0;
            if (DYNAMIC != 0){
                bloom_disabled_flag[i] = -1;
            }
            else{
                bloom_disabled_flag[i] = 1;
            }
        } else {
            bl_ptrs[i] = NULL;
            bloom_cnters[i] = 0;
            bloom_disabled_flag[i] = -1;
        }
    }
    pg_lip_bloom_make_shared();
    for (i = 0; i < n_bloom_used; i++){
        pg_lip_bloom_make_local(i);
    }
    PG_RETURN_INT32(0);
}

/*******************************************************************
* Function: pg_lip_bloom_add()
* Description: add value to the bloom filter
* Parameters:
        int bl_idx - the bloom filter index
        int val - the value to add
* Return:
        int ret - 0/1 whether the added key has a collision in the bloom filter
*******************************************************************/
PG_FUNCTION_INFO_V1(pg_lip_bloom_add);
Datum
pg_lip_bloom_add(PG_FUNCTION_ARGS)
{
    int bl_idx = PG_GETARG_INT32(0);
    int32 val = PG_GETARG_INT32(1);
    int ret;
    struct bloom *bloom_ptr;
    if (bl_idx < 0 || bl_idx >= n_bloom_used)
        ereport(ERROR,
                (errmsg("Bloom filter index %d is out of range [0, %d)",
                        bl_idx, n_bloom_used)));
    if ( bl_ptrs[bl_idx] == NULL ){
        pg_lip_bloom_make_local(bl_idx);
    }
    bloom_ptr = bl_ptrs[bl_idx];
    ret = bloom_add(bloom_ptr, &val, sizeof(int32));
    bloom_cnters[bl_idx] += 1;
    PG_RETURN_INT32(ret);
}

/*******************************************************************
* Function: _bloom_probe()
* Description: the helper function for bloom filter probing
* Parameters:
        int bl_idx - the bloom filter index
        int val - the value to be probed
* Return:
        boolean ret - whether the added key has a collision in the bloom filter
*******************************************************************/
static bool _bloom_probe(int bl_idx, int32 val){

    bool ret = true;
    if (bl_ptrs[bl_idx] == NULL) {
        pg_lip_bloom_make_local(bl_idx);
    }

    if (bloom_disabled_flag[bl_idx]==-1) {
        struct bloom *bloom_ptr  = bl_ptrs[bl_idx];
        bloom_ptr -> probe_cnt += 1;
        ret = bloom_check(bloom_ptr, &val, sizeof(int32));
        if (ret == false){
            bloom_ptr -> prune_cnt += 1;
        }
        if (bloom_ptr -> probe_cnt >= DYNMIAC_DETECT_FREQ){
            if (bloom_ptr -> prune_cnt < DYNMIAC_DETECT_FREQ * MIN_ACTIVATE_PRUNE_RATE){
                bloom_disabled_flag[bl_idx] = 0;
                bloom_ptr -> stale_timer = DYNMIAC_DETECT_FREQ * 10;
                // elog(NOTICE, "Bloom filter #%d is disabled. Pruned: [%.2f] percent ", bl_idx, 100 * ((float)(bloom_ptr->prune_cnt) / bloom_ptr -> probe_cnt));
            }
            else {
                bloom_disabled_flag[bl_idx] = 1;
                bloom_ptr -> stale_timer = DYNMIAC_DETECT_FREQ * 10;
                // elog(NOTICE, "Bloom filter #%d is enabled. Pruned: [%.2f] percent ", bl_idx, 100 * ((float)(bloom_ptr->prune_cnt) / bloom_ptr -> probe_cnt));
            }
        }
    }
    else if (bloom_disabled_flag[bl_idx]==1) {
        ret = bloom_check(bl_ptrs[bl_idx], &val, sizeof(int32));
        if (DYNAMIC == 2){
            bl_ptrs[bl_idx] -> stale_timer -= 1;
            if( bl_ptrs[bl_idx] -> stale_timer <= 0){
                // reset the probe count
                bl_ptrs[bl_idx] -> probe_cnt = 0;
                bl_ptrs[bl_idx] -> prune_cnt = 0;
                bloom_disabled_flag[bl_idx] = -1;
            }
        }
        else if (DYNAMIC == 0) {
            bl_ptrs[bl_idx] -> probe_cnt += 1;
            if (ret == false){
                bl_ptrs[bl_idx] -> prune_cnt += 1;
            }
        }
    }
    else if (bloom_disabled_flag[bl_idx]==0) {
        if (DYNAMIC == 2){
            bl_ptrs[bl_idx] -> stale_timer -= 1;
            if( bl_ptrs[bl_idx] -> stale_timer <= 0){
                bloom_disabled_flag[bl_idx] = -1;
            }
        }
        ret = true;
    }
    return ret;
}

/*******************************************************************
* Function: pg_lip_bloom_probe()
* Description: the PG extension function for bloom filter probing
* Parameters:
        int bl_idx - the bloom filter index
        int val - the value to be probed
* Return:
        boolean ret - whether the added key has a collision in the bloom filter
*******************************************************************/
PG_FUNCTION_INFO_V1(pg_lip_bloom_probe);
Datum
pg_lip_bloom_probe(PG_FUNCTION_ARGS)
{
    int32 bl_idx = PG_GETARG_INT32(0);
    int32 val = PG_GETARG_INT32(1);
    bool ret;
    if (bl_idx < 0 || bl_idx >= n_bloom_used)
        ereport(ERROR,
                (errmsg("Bloom filter index %d is out of range [0, %d)",
                        bl_idx, n_bloom_used)));
    ret = _bloom_probe(bl_idx, val);
    PG_RETURN_BOOL(ret);
}

/*******************************************************************
* Function: pg_lip_bloom_probe()
* Description: the PG extension function that prints out the states of the bloom filters
* Parameters: void
* Return: void
*******************************************************************/
PG_FUNCTION_INFO_V1(pg_lip_bloom_info);
Datum
pg_lip_bloom_info(PG_FUNCTION_ARGS)
{
    int i;
    int32 total_probes = 0, duplicate_probes = 0;
    elog(NOTICE, "Current DYNAMIC setting: %d", DYNAMIC) ;
    for (i = 0; i < n_bloom_used; i++){
        float prune_rate = 0.0;
        if (bl_ptrs[i] == NULL)
            continue;
        if (bl_ptrs[i]->probe_cnt > 0)
            prune_rate = bl_ptrs[i]->prune_cnt / bl_ptrs[i]->probe_cnt;
        elog(NOTICE, "Bloom #%d at %p with %d elements added. [SIG: %d] [Filtered: %.4f] [Total probe: %d]",   i,
                                                                        bl_ptrs[i],
                                                                        bloom_cnters[i],
                                                                        bloom_get_content_signature(bl_ptrs[i]),
                                                                        prune_rate,
                                                                        (int32)(bl_ptrs[i] -> probe_cnt)
                                                                        );
    }
    elog(NOTICE, "Duplicate probes / Total probes = %d / %d (= %.2f)",
         duplicate_probes, total_probes,
         total_probes > 0 ? (float) duplicate_probes / total_probes : 0.0);
    PG_RETURN_INT32(0);
}
