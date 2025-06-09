#ifndef NR_PREDICT_H
#define NR_PREDICT_H

#include "postgres.h"

typedef struct NeurDBInferenceResult {
    Oid typeoid;
	char *result;
    List *id_class_map;
} NeurDBInferenceResult;

#endif