#ifndef TABULAR_INFERENCE_TASK_H
#define TABULAR_INFERENCE_TASK_H

#include "inference_task.h"

struct TabularClassificationBatchData final : ProcessedBatchData {
    size_t batch_size = 0;
    int64_t num_features = 0;
    std::vector<int64_t> features;
    // std::vector<float> features;

    size_t batchSize() const override { return batch_size; }

    std::vector<std::pair<std::string, Ort::Value> > toOrtValues(Ort::MemoryInfo &memory_info) const override;
};

class TabularClassificationTask final : public InferenceTask {
public:
    TabularClassificationTask(
        std::vector<std::string> input_cols,
        std::string output_col,
        int batch_size
    ): InferenceTask(std::move(input_cols),
        std::move(output_col),
        batch_size) {
    }

    std::unique_ptr<ProcessedBatchData>
    preprocess(const RawBatchData &raw) override;

    std::vector<std::any>
    inference(
        const ProcessedBatchData &proc,
        Ort::Session &sess,
        Ort::MemoryInfo &mem
    ) override;

    float getMetric(
        const std::vector<std::any> &predictions,
        const std::vector<std::any> &ground_truth
    ) const override;
};

#endif //TABULAR_INFERENCE_TASK_H
