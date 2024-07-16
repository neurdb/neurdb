#ifndef HTTP_H
#define HTTP_H

/**
 * Resquest the server to train a model
 * @param libsvm_data char* Training data in libsvm format
 * @param batch_size int Batch size in training
 * @param model_name char* Model name
 */
void request_train(const char *libsvm_data, int batch_size, const char *model_name);

/**
 * Resquest the server to make a forward inference with a model
 * @param libsvm_data char* Inference data in libsvm format
 * @param model_name char* Model name
 * @param model_id int Trained model id
 */
void request_inference(const char *libsvm_data, const char *model_name, int model_id);
#endif
