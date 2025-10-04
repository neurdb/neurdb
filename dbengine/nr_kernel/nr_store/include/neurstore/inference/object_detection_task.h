#ifndef OBJECT_DETECTION_TASK_H
#define OBJECT_DETECTION_TASK_H

#include "neurstore/inference/inference_task.h"
#include <opencv2/opencv.hpp>

struct Box {
    float x1, y1, x2, y2;
    int label;
    float score;
};

struct ObjectDetectionBatchData final : ProcessedBatchData {
    size_t batch_size;
    std::vector<float> pixel_values; // [B, 3, H, W]
    std::vector<int64_t> orig_heights;
    std::vector<int64_t> orig_widths;
    int img_h{0}, img_w{0};

    size_t batchSize() const override { return batch_size; }

    std::vector<std::pair<std::string, Ort::Value> >
    toOrtValues(Ort::MemoryInfo &mem) const override;
};

class ObjectDetectionTask final : public InferenceTask {
public:
    explicit ObjectDetectionTask(
        std::vector<std::string> input_cols,
        std::string output_col,
        int batch_size,
        int img_h = 400,
        int img_w = 400,
        float score_threshold = 0.1f,
        float nms_iou_threshold = 0.5f
    ) : InferenceTask(std::move(input_cols), std::move(output_col), batch_size)
        , img_h_(img_h)
        , img_w_(img_w)
        , score_thresh_(score_threshold)
        , nms_iou_thresh_(nms_iou_threshold) {
    }

    std::unique_ptr<ProcessedBatchData>
    preprocess(const RawBatchData &raw) override;

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
    static std::vector<float> preprocessImage(
        const std::string &path,
        int h,
        int w,
        int64_t &orig_h,
        int64_t &orig_w
    );

    static float IoU(const Box &a, const Box &b);

    std::vector<Box> nms(const std::vector<Box> &boxes) const;

    int img_h_, img_w_;
    float score_thresh_, nms_iou_thresh_;
};

#endif // OBJECT_DETECTION_TASK_H
