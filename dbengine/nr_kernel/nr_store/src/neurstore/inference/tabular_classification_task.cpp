#include "neurstore/inference/tabular_classification_task.h"
#include <cmath>
#include <stdexcept>


std::vector<std::pair<std::string, Ort::Value> >
TabularClassificationBatchData::toOrtValues(Ort::MemoryInfo &memory_info) const {
    std::array<int64_t, 2> shape{
        static_cast<int64_t>(batch_size),
        num_features
    };

    Ort::Value X = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(features.data()),
        features.size(),
        shape.data(), shape.size()
    );

    std::vector<std::pair<std::string, Ort::Value> > res;
    res.emplace_back("input", std::move(X));
    return res;
}

std::unique_ptr<ProcessedBatchData>
TabularClassificationTask::preprocess(const RawBatchData &raw) {
    std::vector<const RawColumnData *> cols;
    for (const auto &name: input_columns) {
        const RawColumnData *ptr = nullptr;
        for (const auto &c: raw.columns)
            if (c.name == name) {
                ptr = &c;
                break;
            }
        if (!ptr)
            throw std::runtime_error("Column \"" + name + "\" not found");
        cols.push_back(ptr);
    }

    size_t B = cols.front()->string_data.size()
               + cols.front()->int64_data.size()
               + cols.front()->float_data.size();
    if (B == 0) B = cols.front()->int32_data.size();

    int64_t F = cols.size();
    auto batch = std::make_unique<TabularClassificationBatchData>();
    batch->batch_size = B;
    batch->num_features = F;
    batch->features.resize(B * F);

    for (int64_t c = 0; c < F; ++c) {
        const auto *col = cols[c];
        for (size_t r = 0; r < B; ++r) {
            float value;
            switch (col->type) {
                case ColumnType::Float: value = col->float_data[r];
                    break;
                case ColumnType::Int: value = static_cast<float>(col->int64_data[r]);
                    break;
                case ColumnType::String:
                    value = std::stof(col->string_data[r]);
                    break;
                default:
                    throw std::runtime_error("Unsupported column type");
            }
            batch->features[r * F + c] = value;
        }
    }
    return batch;
}

std::vector<std::any>
TabularClassificationTask::inference(
    const ProcessedBatchData &processed_data,
    Ort::Session &session,
    Ort::MemoryInfo &memory_info
) {
    const auto &batch = dynamic_cast<const TabularClassificationBatchData &>(processed_data);
    auto inputs = batch.toOrtValues(memory_info);

    std::vector<const char *> in_names;
    std::vector<Ort::Value> in_vals;
    for (auto &[n, v]: inputs) {
        in_names.push_back(n.c_str());
        in_vals.emplace_back(std::move(v));
    }

    const char *out_names[] = {"logit"};

    auto outs = session.Run(
        Ort::RunOptions{nullptr},
        in_names.data(),
        in_vals.data(),
        in_vals.size(),
        out_names,
        1
    );

    const float *logits = outs[0].GetTensorData<float>();
    int64_t B = batch.batch_size;

    std::vector<std::any> results;
    results.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        float p1 = 1.f / (1.f + std::exp(-logits[i]));
        int pred = p1 >= 0.5f;
        results.emplace_back(pred);
    }
    return results;
}

float TabularClassificationTask::getMetric(
    const std::vector<std::any> &predictions,
    const std::vector<std::any> &ground_truth
) const {
    size_t correct = 0;
    size_t total = std::min(predictions.size(), ground_truth.size());
    for (size_t i = 0; i < total; ++i) {
        int64_t prediction = std::any_cast<int64_t>(predictions[i]);
        int64_t label = std::any_cast<int64_t>(ground_truth[i]);
        if (prediction == label) correct++;
    }
    return static_cast<float>(correct) / total;
}
