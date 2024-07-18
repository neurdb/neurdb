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
 * @param batch_size int Batch size in inference, 0 for single inference
 */
void request_inference(const char *libsvm_data, const char *model_name, int model_id, int batch_size);

/**
 * Resquest the server to finetune a model
 * @param libsvm_data char* Finetune data in libsvm format
 * @param model_name char* Model name
 * @param model_id int Trained model id
 * @param batch_size int Batch size in finetune
 */
void request_finetune(const char *libsvm_data, const char *model_name, int model_id, int batch_size);
#endif