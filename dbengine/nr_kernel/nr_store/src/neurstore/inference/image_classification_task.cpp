#include "neurstore/inference/image_classification_task.h"

#include <algorithm>
#include <stdexcept>

std::vector<std::pair<std::string, Ort::Value> >
ImageClassificationBatchData::toOrtValues(Ort::MemoryInfo &mem) const {
    std::array<int64_t, 4> dims = {
        static_cast<int64_t>(batch_size),
        3,
        height,
        width
    };

    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float *>(pixel_values.data()),
        pixel_values.size(),
        dims.data(),
        dims.size()
    );

    std::vector<std::pair<std::string, Ort::Value>> result;
    result.emplace_back("pixel_values", std::move(tensor));
    return result;
}

std::unique_ptr<ProcessedBatchData>
ImageClassificationTask::preprocess(const RawBatchData &raw) {
    const RawColumnData *col_img = nullptr;
    for (const auto &col: raw.columns) {
        if (col.name == input_columns[0] && col.type == ColumnType::String) {
            // we assume the first input column is the one containing image paths
            col_img = &col;
            break;
        }
    }
    if (!col_img) throw std::runtime_error("Image path column not found");

    size_t B = col_img->string_data.size();
    auto batch = std::make_unique<ImageClassificationBatchData>();
    batch->batch_size = B;
    batch->height = img_h_;
    batch->width = img_w_;
    batch->pixel_values.resize(B * 3 * img_h_ * img_w_);

    for (size_t i = 0; i < B; ++i) {
        // channel-height-width order
        auto chw = preprocessImage(col_img->string_data[i], img_h_, img_w_);
        std::copy(
            chw.begin(),
            chw.end(),
            batch->pixel_values.begin() + i * chw.size()
        );
    }
    return batch;
}

std::vector<std::any>
ImageClassificationTask::inference(
    const ProcessedBatchData &processed_data,
    Ort::Session &session,
    Ort::MemoryInfo &memory_info
) {
    const auto &batch = dynamic_cast<const ImageClassificationBatchData &>(processed_data);
    auto inputs = batch.toOrtValues(memory_info);

    std::vector<const char *> inp_names, out_names = {"logits"};
    std::vector<Ort::Value> inp_vals;
    inp_names.reserve(inputs.size());
    for (auto &[n, v]: inputs) {
        inp_names.push_back(n.c_str());
        inp_vals.emplace_back(std::move(v));
    }

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inp_names.data(), inp_vals.data(), inp_vals.size(),
        out_names.data(), 1
    );

    const float *logits = outputs[0].GetTensorData<float>();
    const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t B = shape[0], C = shape[1];

    std::vector<std::any> results;
    results.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        const float *row = logits + i * C;
        int cls = std::distance(row, std::max_element(row, row + C));
        results.emplace_back(cls);
    }
    return results;
}

std::vector<float>
ImageClassificationTask::preprocessImage(const std::string &path,
                                         int h, int w) {
    cv::Mat img = cv::imread(path); // BGR
    if (img.empty())
        throw std::runtime_error("Failed to read image: " + path);

    cv::resize(img, img, cv::Size(w, h));
    img.convertTo(img, CV_32F, 1.0 / 255);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // we assume the model expects normalized pixel values
    static const float mean[3] = {0.5f, 0.5f, 0.5f};
    static const float stdv[3] = {0.5f, 0.5f, 0.5f};

    std::vector<float> chw(3 * h * w);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float v = img.at<cv::Vec3f>(y, x)[c];
                chw[c * h * w + y * w + x] = (v - mean[c]) / stdv[c];
            }
    return chw;
}

float ImageClassificationTask::getMetric(
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
