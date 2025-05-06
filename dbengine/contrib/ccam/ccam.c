#include "postgres.h"
#include "fmgr.h"
#include "access/tableam.h"
// #include "utils/guc.h"

PG_MODULE_MAGIC;

void _PG_init(void);
void _PG_fini(void);

Datum ccam_tableam_handler(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(ccam_tableam_handler);

static const TableAmRoutine ccam_methods = {
    .type = T_TableAmRoutine,
    .slot_callbacks = table_slot_callbacks, // reuse heap slot for now
    .scan_begin = NULL,                      // you'll implement this
    .scan_getnextslot = NULL,
    .tuple_insert = NULL,
    .tuple_update = NULL,
    .tuple_delete = NULL,
    // other methods as needed
};

Datum
ccam_tableam_handler(PG_FUNCTION_ARGS)
{
    PG_RETURN_POINTER(&ccam_methods);
}

void
_PG_init(void)
{
    // Optional: GUCs here
}
