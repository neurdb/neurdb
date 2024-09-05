#ifndef STORAGE_H
#define STORAGE_H

#include <stddef.h>

#include "../utils/torch/torch_wrapper.h"

/**
 * Layer storage, this is used to store the layer information in the database
 */
typedef struct {
  char *layer_class;
  char *init_params;
  void *state_dict;
  size_t state_dict_size;
  char *name;
} LayerStorage;

/**
 * Create a layer storage
 * @param layer_class the class of the layer, e.g. "torch.nn.Linear"
 * @param init_params the initialization parameters of the layer, e.g.
 * "in_features=10, out_features=5"
 * @param state_dict the state dictionary of the layer
 * @param state_dict_size the size of the state dictionary
 * @param name the name of the layer, this is optional, pass NULL if not needed
 * @return
 */
LayerStorage *create_layer_storage(const char *layer_class,
                                   const char *init_params, void *state_dict,
                                   size_t state_dict_size, const char *name);

/**
 * Serialize the layer storage into a byte array
 * @param layer_storage layer storage to be serialized
 * @param serialized_length the length of the serialized byte array
 * @return the serialized byte array
 */
void *serialize_layer_storage(const LayerStorage *layer_storage,
                              size_t *serialized_length);

/**
 * Deserialize the byte array into a layer storage
 * @param data the byte array in byte array
 * @param length the length of the byte array
 * @return the deserialized layer storage object
 */
void *deserialize_layer_storage(const void *data, size_t length);

/**
 * Convert a layer storage to a model wrapper, this can be done since layers in
 * PyTorch are also nn.Modules
 * @param layer_storage the layer storage to be converted
 * @return the model wrapper
 */
ModelWrapper *to_model_wrapper(const LayerStorage *layer_storage);

/**
 * Free the layer storage
 * @param layer_storage the layer storage to be freed
 */
void free_layer_storage(LayerStorage *layer_storage);

/**
 * ModelStorage is used to store the model
 */
typedef struct {
  char *model_class;
  char *init_params;
  LayerStorage **layers;
  size_t layer_count;
} ModelStorage;

/**
 * PickledModelStorage is used to store the pickled model, this is used to store
 * the model in the database
 */
typedef struct {
  void *model_meta_pickled;
  size_t model_meta_pickled_size;
  void **layer_sequence_pickled;
  size_t *layer_sequence_pickled_size;
  size_t layer_count;
} PickledModelStorage;

/**
 * Create a model storage
 * @param model_class the class of the model, e.g. "torch.nn.Sequential"
 * @param init_params the initialization parameters of the model, e.g.
 * "layers=[torch.nn.Linear(10, 5), torch.nn.ReLU()]"
 * @param layers the layers of the model
 * @param layer_count the number of layers
 * @return the model storage
 */
ModelStorage *create_model_storage(const char *model_class,
                                   const char *init_params,
                                   LayerStorage **layers, size_t layer_count);

/**
 * Create a model storage from a model
 * @param model the model to be converted
 * @return the model storage
 */
ModelStorage *from_model(const ModelWrapper *model);

/**
 * Append a layer to the model storage
 * @param model_storage the model storage
 * @param layer_storage the layer storage
 */
void append_layer(ModelStorage *model_storage, LayerStorage *layer_storage);

/**
 * Convert a model storage to a ModelWrapper
 * @param model_storage the model storage to be converted
 * @return the model wrapper
 */
ModelWrapper *to_model(const ModelStorage *model_storage);

/**
 * Serialize the model storage
 * @param model_storage the model storage to be serialized
 * @return the pickled model storage
 */
PickledModelStorage *serialize_model_storage(const ModelStorage *model_storage);

/**
 * Deserialize the pickled model storage
 * @param pickled_model_storage  the pickled model storage to be deserialized
 * @return the model storage
 */
ModelStorage *deserialize_model_storage(
    const PickledModelStorage *pickled_model_storage);

/**
 * Free the model storage
 * @param model_storage the model storage to be freed
 */
void free_model_storage(ModelStorage *model_storage);

/**
 * Free the pickled model storage
 * @param pickled_model_storage the pickled model storage to be freed
 */
void free_pickled_model_storage(PickledModelStorage *pickled_model_storage);

#endif  // STORAGE_H
