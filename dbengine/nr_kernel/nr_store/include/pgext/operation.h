#ifndef PG_EXT_OPERATION_H
#define PG_EXT_OPERATION_H

#include <postgres.h>
#include <access/heapam.h>


Relation pgext_open(const char *table, LOCKMODE lockmode);

void pgext_close(Relation rel, LOCKMODE lockmode);

void pgext_update_index(Relation rel, TupleTableSlot *slot);

void pgext_select(
    Relation rel,
    Oid index_oid,
    ScanKey keys,
    int nkeys,
    void (*callback)(TupleTableSlot *slot, void *arg),
    void *callback_arg
);

void pgext_insert(Relation rel, Datum *values, bool *nulls);

void pgext_insert_many(Relation rel, Datum **values_array, bool **nulls_array, int nrows);

void pgext_save_to_file(const char *directory, const char *filename, const char *data, size_t size);

Datum pgext_nextval(const char *seqname);

#endif //PG_EXT_OPERATION_H
