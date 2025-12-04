#ifndef IMAGE_CLASSIFICATION_TASK_H
#define IMAGE_CLASSIFICATION_TASK_H

#include "neurstore/inference/inference_task.h"
#include <opencv2/opencv.hpp>


struct ImageClassificationBatchData final : ProcessedBatchData {
    size_t batch_size;
    int64_t height;
    int64_t width;
    std::vector<float> pixel_values; // [B, 3, H, W]

    size_t batchSize() const override { return batch_size; }

    std::vector<std::pair<std::string, Ort::Value> > toOrtValues(Ort::MemoryInfo &mem) const override;
};

class ImageClassificationTask final : public InferenceTask {
public:
    ImageClassificationTask(
        std::vector<std::string> input_cols,
        std::string output_col,
        int batch_size,
        int img_h = 224,
        int img_w = 224
    ): InferenceTask(std::move(input_cols),
                     std::move(output_col),
                     batch_size),
       img_h_(img_h), img_w_(img_w) {
    }

    std::unique_ptr<ProcessedBatchData> preprocess(const RawBatchData &raw) override;

    std::vector<std::any> inference(
        const ProcessedBatchData &proc,
        Ort::Session &sess,
        Ort::MemoryInfo &mem
    ) override;

    float getMetric(
        const std::vector<std::any> &predictions,
        const std::vector<std::any> &ground_truth
    ) const override;

private:
    static std::vector<float> preprocessImage(const std::string &path, int h, int w);
    int img_h_;
    int img_w_;
};

#endif //IMAGE_CLASSIFICATION_TASK_H
