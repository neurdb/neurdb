#include "neurstore/inference/tabular_regression_task.h"

std::vector<std::pair<std::string, Ort::Value> >
TabularRegressionBatchData::toOrtValues(Ort::MemoryInfo &memory_info) const {
    std::array<int64_t, 2> dims{
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(n_features)
    };

    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(features.data()),
        features.size(),
        dims.data(), dims.size()
    );
    std::vector<std::pair<std::string, Ort::Value> > result;
    result.emplace_back("features", std::move(tensor));
    return result;
}

std::unique_ptr<ProcessedBatchData> TabularRegressionTask::preprocess(const RawBatchData &raw) {
    const size_t D = input_columns.size();
    if (raw.columns.size() != D) {
        throw std::runtime_error("TabularRegressionTask: column count mismatch");
    }

    const size_t B =
            raw.columns[0].int64_data.size() +
            raw.columns[0].float_data.size() +
            raw.columns[0].string_data.size();

    auto batch = std::make_unique<TabularRegressionBatchData>();
    batch->batch_size = B;
    batch->n_features = D;
    batch->features.resize(B * D);

    for (size_t col_id = 0; col_id < D; ++col_id) {
        const auto &column = raw.columns[col_id];
        size_t offset = col_id;
        switch (column.type) {
            case ColumnType::Int:
                for (size_t r = 0; r < B; ++r) {
                    batch->features[offset + r * D] = static_cast<float>(column.int64_data[r]);
                }
                break;
            case ColumnType::Float:
                for (size_t r = 0; r < B; ++r) {
                    batch->features[offset + r * D] = column.float_data[r];
                }
                break;
            case ColumnType::String:
                for (size_t r = 0; r < B; ++r) {
                    batch->features[offset + r * D] = std::stof(column.string_data[r]);
                }
                break;
            default:
                throw std::runtime_error("TabularRegressionTask: unsupported column type");
        }
    }
    return batch;
}

std::vector<std::any> TabularRegressionTask::inference(
    const ProcessedBatchData &processed_data,
    Ort::Session &session,
    Ort::MemoryInfo &memory_info
) {
    const auto &batch = dynamic_cast<const TabularRegressionBatchData &>(processed_data);
    auto inputs = batch.toOrtValues(memory_info);

    std::vector<const char *> input_names{
        inputs[0].first.c_str()
    };
    std::vector<Ort::Value> input_values;
    input_values.emplace_back(std::move(inputs[0].second));

    const char *output_names[] = {"pred"};

    auto outs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_values.data(),
        1,
        output_names,
        1
    );

    const float *p = outs[0].GetTensorData<float>();
    size_t N = batch.batch_size;

    std::vector<std::any> preds;
    preds.reserve(N);
    for (size_t i = 0; i < N; ++i)
        preds.emplace_back(p[i]);

    return preds;
}

float TabularRegressionTask::getMetric(
    const std::vector<std::any>& predictions,
    const std::vector<std::any>& ground_truth
) const {
    size_t N = std::min(predictions.size(), ground_truth.size());
    if (N == 0) return 0.0f;

    double mse = 0.0;
    for (size_t i = 0; i < N; ++i) {
        float pred = std::any_cast<float>(predictions[i]);
        float label = std::any_cast<float>(ground_truth[i]);
        double diff = pred - label;
        mse += diff * diff;
    }
    return static_cast<float>(mse / N);
}
