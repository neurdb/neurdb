#ifndef INFERENCE_TASK_H
#define INFERENCE_TASK_H

#include <any>

#include "neurstore/inference/inference_utils.h"


class InferenceTask {
public:
    InferenceTask(
        std::vector<std::string> input_columns,
        std::string output_column,
        int batch_size
    ) : input_columns(std::move(input_columns)),
        output_column(std::move(output_column)),
        batch_size(batch_size) {}

    virtual ~InferenceTask() = default;

    virtual std::unique_ptr<ProcessedBatchData> preprocess(
        const RawBatchData& raw_data
    ) = 0;

    virtual std::vector<std::any> inference(
        const ProcessedBatchData& processed_data,
        Ort::Session& session,
        Ort::MemoryInfo& memory_info
    ) = 0;

    virtual float getMetric(
        const std::vector<std::any>& predictions,
        const std::vector<std::any>& ground_truth
    ) const = 0;

    std::vector<std::string> input_columns;
    std::string output_column;
    int batch_size;
};

#endif //INFERENCE_TASK_H
