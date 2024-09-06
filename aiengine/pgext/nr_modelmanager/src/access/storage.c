#include "storage.h"

#include <stdlib.h>
#include <string.h>

ModelStorage *create_model_storage(const char *model_class,
                                   const char *init_params,
                                   LayerStorage **layers,
                                   const size_t layer_count) {
  ModelStorage *model_storage = (ModelStorage *)malloc(sizeof(ModelStorage));
  model_storage->model_class = strdup(model_class);
  model_storage->init_params = strdup(init_params);
  model_storage->layers =
      (LayerStorage **)malloc(sizeof(LayerStorage *) * layer_count);
  model_storage->layer_count = layer_count;

  for (size_t i = 0; i < layer_count; i++) {
    // copy the layer storage pointer
    model_storage->layers[i] = layers[i];
  }
  return model_storage;
}

ModelStorage *from_model(const ModelWrapper *model) {
  // TODO: implement this function
  return NULL;
}

void append_layer(ModelStorage *model_storage, LayerStorage *layer_storage) {
  model_storage->layers = (LayerStorage **)realloc(
      model_storage->layers,
      sizeof(LayerStorage *) * (model_storage->layer_count + 1));
  model_storage->layers[model_storage->layer_count] = layer_storage;
  model_storage->layer_count++;  // update layer_count
}

ModelWrapper *to_model(const ModelStorage *model_storage) {
  // TODO: implement this function
  return NULL;
}

PickledModelStorage *serialize_model_storage(
    const ModelStorage *model_storage) {
  // TODO: implement this function
  return NULL;
}

ModelStorage *deserialize_model_storage(
    const PickledModelStorage *pickled_model_storage) {
  // TODO: implement this function
  return NULL;
}

void free_model_storage(ModelStorage *model_storage) {
  free(model_storage->model_class);
  free(model_storage->init_params);
  for (size_t i = 0; i < model_storage->layer_count; i++) {
    free_layer_storage(model_storage->layers[i]);
  }
  free(model_storage->layers);
  free(model_storage);
}

void free_pickled_model_storage(PickledModelStorage *pickled_model_storage) {
  free(pickled_model_storage->model_meta_pickled);
  for (size_t i = 0; i < pickled_model_storage->layer_count; i++) {
    free(pickled_model_storage->layer_sequence_pickled[i]);
  }
  free(pickled_model_storage->layer_sequence_pickled);
  free(pickled_model_storage->layer_sequence_pickled_size);
  free(pickled_model_storage);
}

LayerStorage *create_layer_storage(const char *layer_class,
                                   const char *init_params, void *state_dict,
                                   size_t state_dict_size, const char *name) {
  // TODO: implement this function
  return NULL;
}

void *serialize_layer_storage(const LayerStorage *layer_storage,
                              size_t *serialized_length) {
  // TODO: implement this function
  return NULL;
}

void *deserialize_layer_storage(const void *data, size_t length) {
  // TODO: implement this function
  return NULL;
}

ModelWrapper *to_model_wrapper(const LayerStorage *layer_storage) {
  // TODO: implement this function
  return NULL;
}

void free_layer_storage(LayerStorage *layer_storage) {
  // TODO: implement this function
}
