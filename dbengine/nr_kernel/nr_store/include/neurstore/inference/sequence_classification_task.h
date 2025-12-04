#ifndef SEQUENCE_CLASSIFICATION_TASK_H
#define SEQUENCE_CLASSIFICATION_TASK_H

#include "neurstore/inference/inference_task.h"


struct SeqClassificationBatchData final : ProcessedBatchData {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    int64_t batch_size;
    int64_t max_seq_len;

    size_t batchSize() const override;

    std::vector<std::pair<std::string, Ort::Value> > toOrtValues(Ort::MemoryInfo &memory_info) const override;
};

class SeqClassificationTask final : public InferenceTask {
public:
    SeqClassificationTask(
        std::vector<std::string> input_columns,
        std::string output_column,
        int batch_size,
        std::unique_ptr<tokenizers::Tokenizer> tokenizer,
        int max_seq_len
    ): InferenceTask(std::move(input_columns), std::move(output_column), batch_size),
       tokenizer_(std::move(tokenizer)),
       max_seq_len_(max_seq_len) {}

    std::unique_ptr<ProcessedBatchData> preprocess(const RawBatchData &raw_data) override;

    std::vector<std::any> inference(
        const ProcessedBatchData &processed_data,
        Ort::Session &session,
        Ort::MemoryInfo &memory_info
    ) override;

    float getMetric(
        const std::vector<std::any>& predictions,
        const std::vector<std::any>& ground_truth
    ) const override;

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    int max_seq_len_;
};


#endif //SEQUENCE_CLASSIFICATION_TASK_H
