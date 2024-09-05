/*
 * interface.h
 *    APIs provided by the nr_model extension
 */
#ifndef PG_MODEL_PG_MODEL_H
#define PG_MODEL_PG_MODEL_H

#include <fmgr.h>
#include <postgres.h>

// /**
//  * @description: register a model to the model table
//  * @param {cstring}    model name
//  * @param {cstring}    model path
//  * @return {bool}      true if success, false otherwise
//  */
// Datum
// pgm_register_model(PG_FUNCTION_ARGS);
//
// /**
//  * @description: unregister a model from the model table,
//  * note that the model file will not be deleted from the file system
//  * @param {cstring}    model name
//  * @return {bool}      true if success, false otherwise
//  */
// Datum
// pgm_unregister_model(PG_FUNCTION_ARGS);
//
// /**
//  * @desciption: store a model to the model table
//  * @param {cstring}    model name
//  * @param {cstring}    model path (in .pt file)
//  * @return {bool}      true if success, false otherwise
//  */
// Datum
// pgm_store_model(PG_FUNCTION_ARGS);
//
// /**
//  * @description: get the model id by the model name
//  * @param {cstring}    model name
//  * @return {int}       model id
//  */
// Datum
// pgm_get_model_id_by_name(PG_FUNCTION_ARGS);

/**
 * @description: make a prediction using the model
 * @param {cstring}    model name
 * @param {any...}     input data
 * @return {float4}    prediction result in float4
 * @note in sql, the input is passed as an anyarray type value
 * @see https://www.postgresql.org/docs/9.4/functions-array.html
 * @see https://www.postgresql.org/docs/16/functions-array.html
 * @see https://www.postgresql.org/docs/current/xfunc-c.html
 */
Datum pgm_predict_float4(PG_FUNCTION_ARGS);

/**
 * @description make a prediction using the model by passing a table
 * @param {cstring}    model name
 * @param {int}        batch size
 * @param {cstring}    table name
 * @param {text[]}     column names
 * @return {null}      void
 * This function is temporary and designed for testing purposes
 */
Datum pgm_predict_table(PG_FUNCTION_ARGS);

/**
 * @description: make a prediction using the model
 * @param {cstring}    model name
 * @param {any...}     input data
 * @return {text}      prediction result in text
 */

// TODO: Datum pgm_predict_text(PG_FUNCTION_ARGS);
#endif
