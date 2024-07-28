#ifndef INTERFACE_H
#define INTERFACE_H

#include <postgres.h>
#include <fmgr.h>

typedef struct {
    char *model_name;
    char *table_name;
    char* client_socket_id;
    int batch_size;
    int epoch;
    int train_batch_num;
    int eva_batch_num;
    int test_batch_num;
} TrainingInfo;

typedef struct {
    char *model_name;
    int model_id;
    char *table_name;
    char* client_socket_id;
    int batch_size;
    int batch_num;
} InferenceInfo;

typedef struct {
    char *model_name;
    int model_id;
    char *table_name;
    char* client_socket_id;
    int batch_size;
    int epoch;
    int train_batch_num;
    int eva_batch_num;
    int test_batch_num;
} FinetuneInfo;

/**
 * Preprocess the input data for model inference. It contains the following steps:
 * 1. Extract the input data from the table
 * 2. It checks if all columns' value type are integers
 *     2.1 If not, it performs one-hot encoding on the specified columns (TODO: this is not implemented yet)
 *     2.2 If yes, it continues to the next step
 * 3. It convert the data to libsvm format, and foward the data to the python server
 * @param model_name text The name of the model
 * @param model_id integer The id of the model to be used in the inference
 * @param table_name text The name of the table to be used in the inference
 * @param batch_size integer The batch size of the input data, 0 for single inference
 * @param features text[] Columns to be used in the inference
 * @return Table
 */
Datum nr_inference(PG_FUNCTION_ARGS);


/**
 * Train the model
 * @param model_name int The name of the model to be trained
 * @param table_name text The name of the table to be used in the training
 * @param batch_size int The batch size of the input data
 * @param epochs int The number of epochs
 * @param features text[] Columns to be used in the training
 * @param target text The target column
 * @return void
 */
Datum nr_train(PG_FUNCTION_ARGS);

/**
 * Finetune a model
 * @param model_name text The name of the model to be finetuned
 * @param model_id int The id of the model to be finetuned
 * @param table_name text The name of the table to be used in the finetuning
 * @param batch_size int The batch size of the input data
 * @param epochs int The number of epochs
 * @param features text[] Columns to be used in the finetuning
 * @param target text The target column
 * @return void
 */
Datum nr_finetune(PG_FUNCTION_ARGS);

#endif //INTERFACE_H
