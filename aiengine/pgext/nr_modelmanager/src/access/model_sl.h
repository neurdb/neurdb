/*
 * model_access.h
 *    provide the APIs for model save, load, and delete
 *    interact with the file system and the model table
 */
#ifndef PG_MODEL_MODEL_SL_H
#define PG_MODEL_MODEL_SL_H

#include <stdbool.h>
#include <c.h>

#include "../utils/torch/torch_wrapper.h"

// /**
//  * @description: register a model to the model table
//  * @param {cstring} model_name - the name of the model
//  * @param {cstring} model_path - the path to the model
//  * @return {bool} - true if success, false otherwise
//  * @deprecated temporarily abandoned, store the model into the database from
//  the python side instead
//  */
// bool
// register_model(const char *model_name, const char *model_path);
//
// /**
//  * @description: unregister a model from the model table
//  * @param {cstring} model_name - the name of the model
//  * @return {bool} - true if success, false otherwise
//  * @deprecated temporarily abandoned, delete the model from the database from
//  the python side instead
//  */
// bool
// unregister_model(const char *model_name);
//
// /**
//  * @description: save the model to the file system
//  * @param {cstring} model_name - the name of the model
//  * @param {cstring} save_path - the path to save the model
//  * @param {ModelWrapper*} model - the model to be saved
//  * @return {bool} - true if success, false otherwise
//  * @note this function will serialize the model and save it to the file
//  system
//  * @deprecated temporarily abandoned, store the model into the database from
//  the python side instead
//  */
// bool
// save_model(const char *model_name, const char *save_path, const ModelWrapper
// *model);
//
// /**
//  * @description: store the model to the database
//  * @param {cstring} model_name - the name of the model
//  * @param {ModelWrapper*} model - the model to be stored
//  * @note this function will serialize the model and save it to the database
//  (not the file system)
//  * @return {bool} - true if success, false otherwise
//  * @deprecated temporarily abandoned, store the model into the database from
//  the python side instead
//  */
// bool
// store_model(const char *model_name, const ModelWrapper *model);
//
// /**
//  * @description: Load the model from the file system by model path
//  * @param {cstring} model_path - the path to the model
//  * @return {ModelWrapper*} - the loaded model if success, NULL otherwise
//  */
// ModelWrapper *
// load_model_by_path(const char *model_path);
//
// /**
//  * @description: Load the model from the serialized data
//  * @param model_bytea - the bytea data of the model
//  * @return {ModelWrapper*} - the loaded model if success, NULL otherwise
//  */
// ModelWrapper *
// load_model_by_bytea(bytea *model_bytea);

/**
 * @description: Load the model from the file system by model id
 * @param {int} model_id - the id of the model in the model table
 * @return {ModelWrapper*} - the loaded model if success, NULL otherwise
 */
ModelWrapper* load_model_by_id(int model_id);

// /**
//  * @description: Load the model from the file system by model name
//  * @param {cstring} model_name - the name of the model
//  * @return {ModelWrapper*} - the loaded model if success, NULL otherwise
//  */
// ModelWrapper *
// load_model_by_name(const char *model_name);
//
// /**
//  * @description: get the model id by model name
//  * @param {cstring} model_name
//  * @return {int} - the model id if success, -1 otherwise
//  */
// int
// get_model_id_by_name(const char *model_name);
#endif
