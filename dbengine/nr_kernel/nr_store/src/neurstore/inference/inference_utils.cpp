#include "neurstore/inference/inference_utils.h"

/************************************ C ************************************/
InferenceOptionsC *ioc_create(
    const char *model_id,
    const char *query,
    const char *pg_dsn,
    const char **input_columns,
    int n_input_columns,
    const char *output_column,
    int batch_size, const
    char *tokenizer_path,
    const char *pad_token,
    const char *eos_token,
    const char *bos_token,
    int max_input_len,
    int max_output_len,
    int load_mode,
    int task,
    int use_gpu
) {
    auto *opt = new InferenceOptions();
    opt->model_id       = model_id       ? model_id       : "";
    opt->query          = query          ? query          : "";
    opt->pg_dsn         = pg_dsn         ? pg_dsn         : "";
    opt->output_column  = output_column  ? output_column  : "";
    opt->tokenizer_path = tokenizer_path ? tokenizer_path : "";
    opt->pad_token      = pad_token      ? pad_token      : "";
    opt->eos_token      = eos_token      ? eos_token      : "";
    opt->bos_token      = bos_token      ? bos_token      : "";

    opt->batch_size      = batch_size;
    opt->max_input_len   = max_input_len;
    opt->max_output_len  = max_output_len;
    opt->load_mode       = static_cast<LoadMode>(load_mode);
    opt->task            = static_cast<Task>(task);
    opt->use_gpu         = (use_gpu != 0);

    opt->input_columns.reserve(n_input_columns);
    for (int i = 0; i < n_input_columns; ++i)
        opt->input_columns.emplace_back(input_columns[i] ? input_columns[i] : "");

    return reinterpret_cast<InferenceOptionsC *>(opt);
}


void ioc_destroy(InferenceOptionsC *options) {
    delete reinterpret_cast<InferenceOptions *>(options);
}
