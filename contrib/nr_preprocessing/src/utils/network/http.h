#ifndef HTTP_H
#define HTTP_H

/**
 * Send a training task to the server
 * @param model_name char* Model name
 * @param dataset_name char* Dataset name
 * @param client_socket_id char* Client socket id
 * @param batch_size int Batch size in training
 * @param epoch int Number of epochs
 * @param train_batch_num int Number of training batches
 * @param eva_batch_num int Number of evaluation batches
 * @param test_batch_num int Number of testing batches
 */
void send_train_task(
    const char *model_name,
    const char *dataset_name,
    const char *client_socket_id,
    int batch_size,
    int epoch,
    int train_batch_num,
    int eva_batch_num,
    int test_batch_num
);

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
