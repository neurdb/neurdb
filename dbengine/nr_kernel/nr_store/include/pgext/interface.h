#ifndef INTERFACE_H
#define INTERFACE_H

#include <postgres.h>
#include <fmgr.h>


/**
 * Save a model to the database
 * @param model_name char* The name of the model
 * @param tolerance float The tolerance during compression
 * @param model_path char* The path to the model file (in .ONNX format)
 * @return bool True if the model is saved successfully, false otherwise
 */
Datum
ns_save_model(PG_FUNCTION_ARGS);

/**
 * Save a model to the database
 * @param model_names char** The names of the models
 * @param tolerance float The tolerance during compression
 * @param model_path char* The folder path to the model files (in .ONNX format)
 * @return bool True if the model is saved successfully, false otherwise
 */
Datum
ns_save_models(PG_FUNCTION_ARGS);

/**
 *
 * Save all models in a folder
 * @param folder_path char* Path to the folder
 * @param tolerance float The tolerance during compression
 * @return bool True if the model is saved successfully, false otherwise
 */
Datum
ns_save_models_from_folder(PG_FUNCTION_ARGS);

/**
 * Save a model to the database without actually saving it,
 * but only to check the compression ratio and performance delta.
 */
Datum
ns_save_model_dry_run(PG_FUNCTION_ARGS);

/**
 * Load a model from the database - full decompression
 * @param model_id int The id of the model
 * @return bytea The decompressed serialized model (in .ONNX format)
 */
Datum
ns_load_model(PG_FUNCTION_ARGS);

/**
 * Load a model from the database - 8-bit base only
 * @param model_id int The id of the model
 * @return bytea The decompressed serialized model (in .ONNX format)
 */
Datum
ns_load_model_as_uint8(PG_FUNCTION_ARGS);

/**
 * Load a model from the database - 8-bit base + delta
 * @param model_id int The id of the model
 * @return bytea The decompressed serialized model (in .ONNX format)
 */
Datum
ns_load_model_as_uint8_delta(PG_FUNCTION_ARGS);

/**
 * Load a model from the database - float16
 * @param model_id int The id of the model
 * @return bytea The decompressed serialized model (in .ONNX format)
 */
Datum
ns_load_model_as_float16(PG_FUNCTION_ARGS);

/**
 * Set the omp parallelism
 * @param parallelism int The number of threads to use for compression
 * @return bool True if the parallelism is set successfully, false otherwise
 */
// Datum
// ns_set_parallelism(PG_FUNCTION_ARGS);

/**
 * Get the current omp parallelism
 * @param parallelism int The number of threads to use for compression
 * @return int The number of threads to use for compression
 */
// Datum
// ns_get_parallelism(PG_FUNCTION_ARGS);

PGDLLEXPORT void bgw_neurstore_main();

Datum
ns_inference(PG_FUNCTION_ARGS);

#endif //INTERFACE_H
