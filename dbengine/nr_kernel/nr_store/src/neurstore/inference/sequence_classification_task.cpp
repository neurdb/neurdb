#include "neurstore/inference/sequence_classification_task.h"

size_t SeqClassificationBatchData::batchSize() const {
    return batch_size;
}

std::vector<std::pair<std::string, Ort::Value> > SeqClassificationBatchData::toOrtValues(
    Ort::MemoryInfo &memory_info) const {
    int64_t dims[2] = {batch_size, max_seq_len};

    Ort::Value ids = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(input_ids.data()),
        input_ids.size(),
        dims,
        2
    );

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(attention_mask.data()),
        attention_mask.size(),
        dims,
        2
    );

    Ort::Value type_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(token_type_ids.data()),
        token_type_ids.size(),
        dims,
        2
    );

    std::vector<std::pair<std::string, Ort::Value>> result;
    result.emplace_back("input_ids", std::move(ids));
    result.emplace_back("attention_mask", std::move(mask_tensor));
    result.emplace_back("token_type_ids", std::move(type_tensor));
    return result;
}

std::unique_ptr<ProcessedBatchData> SeqClassificationTask::preprocess(const RawBatchData &raw_data) {
    const RawColumnData* input_col = nullptr;

    for (const auto& col : raw_data.columns) {
        if (col.name == input_columns[0] && col.type == ColumnType::String) {
            // we make the assumption that for sequence classification tasks, the first input column is
            // the one which contains the text data to classify.
            input_col = &col;
            break;
        }
    }
    if (!input_col) {
        throw std::runtime_error("Input column not found or not of type String");
    }

    const auto& lines = input_col->string_data;
    size_t B = lines.size();

    std::vector<std::vector<int64_t>> all_token_ids(B);
    size_t max_len = 0;

    for (size_t i = 0; i < B; ++i) {
        auto ids_u32 = tokenizer_->Encode(lines[i]);
        if (ids_u32.size() > static_cast<size_t>(max_seq_len_)) ids_u32.resize(max_seq_len_);
        all_token_ids[i] = std::vector<int64_t>(ids_u32.begin(), ids_u32.end());
        max_len = std::max(max_len, all_token_ids[i].size());
    }

    auto batch = std::make_unique<SeqClassificationBatchData>();
    batch->batch_size = B;
    batch->max_seq_len = static_cast<int64_t>(max_len);
    batch->input_ids.resize(B * max_len, 0);
    batch->attention_mask.resize(B * max_len, 0);
    batch->token_type_ids.resize(B * max_len, 0);

    for (size_t i = 0; i < B; ++i) {
        const auto& ids = all_token_ids[i];
        size_t offset = i * max_len;
        std::copy(
            ids.begin(),
            ids.end(),
            batch->input_ids.begin() + offset
        );
        std::fill(
            batch->attention_mask.begin() + offset,
            batch->attention_mask.begin() + offset + ids.size(),
            1
        );
        // for token_type_ids, we assume a single sequence input, so they are all zeros.
    }
    return batch;
}

std::vector<std::any> SeqClassificationTask::inference(
    const ProcessedBatchData &processed_data,
    Ort::Session &session,
    Ort::MemoryInfo &memory_info
) {
    const auto& batch = dynamic_cast<const SeqClassificationBatchData&>(processed_data);
    auto ort_inputs = batch.toOrtValues(memory_info);

    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_values;
    for (auto& [name, val] : ort_inputs) {
        input_names.push_back(name.c_str());
        input_values.push_back(std::move(val));
    }

    const char* output_names[] = {"logits"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_values.data(), input_values.size(),
        output_names, 1
    );

    const float* logits = outputs[0].GetTensorData<float>();
    const auto& shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t B = shape[0], C = shape[1];

    std::vector<std::any> results;
    results.reserve(B);

    for (int64_t i = 0; i < B; ++i) {
        const float* row = logits + i * C;
        int pred = std::distance(row, std::max_element(row, row + C));
        results.emplace_back(pred);
    }
    return results;
}

float SeqClassificationTask::getMetric(
    const std::vector<std::any> &predictions,
    const std::vector<std::any> &ground_truth
) const {
    size_t correct = 0;
    size_t total = std::min(predictions.size(), ground_truth.size());
    for (size_t i = 0; i < total; ++i) {
        int prediction = std::any_cast<int>(predictions[i]);
        int label = std::any_cast<int>(ground_truth[i]);
        if (prediction == label) correct++;
    }
    return static_cast<float>(correct) / total;
}
