#include "pgext/operation.h"

#include <parser/parse_func.h>
#include <access/genam.h>
#include <executor/executor.h>
#include <utils/builtins.h>
#include <utils/snapmgr.h>
#include <utils/lsyscache.h>
#include <sys/stat.h>


Relation pgext_open(const char *table, LOCKMODE lockmode) {
    Oid relOid = RelnameGetRelid(table);
    if (!OidIsValid(relOid)) {
        // Table not found
        ereport(ERROR, (errcode(ERRCODE_UNDEFINED_TABLE), errmsg("table \"%s\" does not exist", table)));
    }
    return relation_open(relOid, lockmode);
}

void pgext_close(Relation rel, LOCKMODE lockmode) {
    relation_close(rel, lockmode);
}

void pgext_update_index(Relation rel, TupleTableSlot *slot) {
    EState *estate = CreateExecutorState();
    ResultRelInfo *resultRelInfo = makeNode(ResultRelInfo);
    InitResultRelInfo(resultRelInfo, rel, 1, NULL, 0);
    ExecOpenIndices(resultRelInfo, false);

    estate->es_opened_result_relations = list_make1(resultRelInfo);
    estate->es_output_cid = GetCurrentCommandId(false);

    List *recheckIndexes = ExecInsertIndexTuples(
        resultRelInfo,
        slot,
        estate,
        false,
        false,
        NULL,
        NIL,
        false
    );

    ExecCloseIndices(resultRelInfo);
    FreeExecutorState(estate);
    list_free(recheckIndexes);
}

void pgext_select(
    Relation rel,
    Oid index_oid,
    ScanKey keys,
    int nkeys,
    void (*callback)(TupleTableSlot *slot, void *arg),
    void *callback_arg
) {
    if (!OidIsValid(index_oid)) {
        // no index, sequential scan
        TableScanDesc scan = table_beginscan(rel, GetActiveSnapshot(), nkeys, keys);
        TupleTableSlot *slot = MakeSingleTupleTableSlot(RelationGetDescr(rel), &TTSOpsHeapTuple);

        while (table_scan_getnextslot(scan, ForwardScanDirection, slot)) {
            callback(slot, callback_arg);
        }
        table_endscan(scan);
        ExecDropSingleTupleTableSlot(slot);
    } else {
        // index scan
        Relation index = index_open(index_oid, AccessShareLock);
        IndexScanDesc scan = index_beginscan(rel, index, GetActiveSnapshot(), nkeys, 0);
        if (nkeys > 0 && keys != NULL) {
            index_rescan(scan, keys, nkeys, NULL, 0);
        }

        // TODO: check TTSOpsBufferHeapTuple, TTSOpsHeapTuple and TTSOpsMinimalTuple
        TupleTableSlot *slot = MakeSingleTupleTableSlot(RelationGetDescr(rel), &TTSOpsBufferHeapTuple);

        while (index_getnext_slot(scan, ForwardScanDirection, slot)) {
            callback(slot, callback_arg);
        }
        index_endscan(scan);
        index_close(index, AccessShareLock);
        ExecDropSingleTupleTableSlot(slot);
    }
}

void pgext_insert(Relation rel, Datum *values, bool *nulls) {
    TupleDesc tupdesc = RelationGetDescr(rel);
    TupleTableSlot *slot = MakeSingleTupleTableSlot(tupdesc, &TTSOpsHeapTuple);
    ExecClearTuple(slot);

    for (int i = 0; i < tupdesc->natts; i++) {
        slot->tts_values[i] = values[i];
        slot->tts_isnull[i] = nulls[i];
    }

    ExecStoreVirtualTuple(slot);
    table_tuple_insert(rel, slot, GetCurrentCommandId(true), 0, NULL);
    pgext_update_index(rel, slot); // update index

    CommandCounterIncrement();
    ExecDropSingleTupleTableSlot(slot);
}

void pgext_insert_many(Relation rel, Datum **values_array, bool **nulls_array, int nrows) {
    TupleDesc tupdesc = RelationGetDescr(rel);
    TupleTableSlot *slot = MakeSingleTupleTableSlot(tupdesc, &TTSOpsHeapTuple);

    EState *estate = CreateExecutorState();
    ResultRelInfo *resultRelInfo = makeNode(ResultRelInfo);
    InitResultRelInfo(resultRelInfo, rel, 1, NULL, 0);
    ExecOpenIndices(resultRelInfo, false);

    estate->es_opened_result_relations = list_make1(resultRelInfo);
    estate->es_output_cid = GetCurrentCommandId(false);

    for (int i = 0; i < nrows; ++i) {
        ExecClearTuple(slot);
        for (int j = 0; j < tupdesc->natts; j++) {
            slot->tts_values[j] = values_array[i][j];
            slot->tts_isnull[j] = nulls_array[i][j];
        }
        ExecStoreVirtualTuple(slot);
        table_tuple_insert(rel, slot, GetCurrentCommandId(true), 0, NULL);
        List *recheck = ExecInsertIndexTuples(
            resultRelInfo,
            slot,
            estate,
            false,
            false,
            NULL,
            NIL,
            false
        );
        list_free(recheck);
    }
    ExecCloseIndices(resultRelInfo);
    ExecDropSingleTupleTableSlot(slot);
    FreeExecutorState(estate);
    CommandCounterIncrement();
}

Datum pgext_nextval(const char *seqname) {
    List *func_name = list_make1(makeString("nextval"));
    Oid arg_types[1] = {REGCLASSOID};
    Oid nextval_oid = LookupFuncName(func_name, 1, arg_types, false);
    list_free(func_name);

    if (!OidIsValid(nextval_oid)) {
        elog(ERROR, "Could not find function nextval(regclass)");
    }
    Oid seqoid = get_relname_relid(seqname, get_namespace_oid("public", false));
    if (!OidIsValid(seqoid)) {
        elog(ERROR, "Sequence \"%s\" not found in schema \"public\"", seqname);
    }
    Datum nextval = OidFunctionCall1(nextval_oid, ObjectIdGetDatum(seqoid));
    return nextval;
}
