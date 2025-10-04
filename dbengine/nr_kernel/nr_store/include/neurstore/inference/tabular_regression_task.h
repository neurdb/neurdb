#ifndef TABULAR_REGRESSION_TASK_H
#define TABULAR_REGRESSION_TASK_H

#include "neurstore/inference/inference_task.h"

struct TabularRegressionBatchData final : ProcessedBatchData {
    size_t batch_size{0};
    size_t n_features{0};
    std::vector<float> features; // [B × D] row‑major

    size_t batchSize() const override {
        return batch_size;
    }

    std::vector<std::pair<std::string, Ort::Value> > toOrtValues(Ort::MemoryInfo &memory_info) const override;
};


class TabularRegressionTask final : public InferenceTask {
public:
    explicit TabularRegressionTask(
        std::vector<std::string> input_columns,
        std::string output_column,
        int batch_size
    ): InferenceTask(std::move(input_columns), std::move(output_column), batch_size) {
    }

    std::unique_ptr<ProcessedBatchData> preprocess(const RawBatchData &raw_data) override;

    std::vector<std::any> inference(
        const ProcessedBatchData &processed_data,
        Ort::Session &session,
        Ort::MemoryInfo &memory_info
    ) override;

    float getMetric(
        const std::vector<std::any> &predictions,
        const std::vector<std::any> &ground_truth
    ) const override;
};

#endif // TABULAR_REGRESSION_TASK_H
