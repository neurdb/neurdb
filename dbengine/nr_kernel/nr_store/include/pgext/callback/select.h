#ifndef CALLBACK_SELECT_H
#define CALLBACK_SELECT_H

#include <postgres.h>
#include <executor/executor.h>


typedef struct ModelSelectResult {
    bool found;
    int model_id;
    char *model_name;
} ModelSelectResult;

void model_select_callback(TupleTableSlot *slot, void *arg);

#endif //CALLBACK_SELECT_H
