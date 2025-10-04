#include "pgext/callback/select.h"

#include <utils/builtins.h>


void model_select_callback(TupleTableSlot *slot, void *arg) {
    ModelSelectResult *result = (ModelSelectResult *) arg;
    bool isnull;

    /* 1: model_id, 2: model_name */
    Datum modelIdDatum = slot_getattr(slot, 1, &isnull);
    Datum modelNameDatum = slot_getattr(slot, 2, &isnull);
    result->model_name = TextDatumGetCString(modelNameDatum);
    result->model_id = DatumGetInt32(modelIdDatum);
    result->found = true;
}
