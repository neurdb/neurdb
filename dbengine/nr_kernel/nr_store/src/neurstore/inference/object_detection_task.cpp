#include "neurstore/inference/object_detection_task.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>


std::vector<std::pair<std::string, Ort::Value> >
ObjectDetectionBatchData::toOrtValues(Ort::MemoryInfo &mem) const {
    std::array<int64_t, 4> dims = {
        static_cast<int64_t>(batch_size),
        3,
        static_cast<int64_t>(img_h),
        static_cast<int64_t>(img_w)
    };

    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float *>(pixel_values.data()),
        pixel_values.size(),
        dims.data(),
        dims.size()
    );

    std::vector<std::pair<std::string, Ort::Value> > res;
    res.emplace_back("pixel_values", std::move(tensor));
    return res;
}

std::vector<float> ObjectDetectionTask::preprocessImage(
    const std::string &path,
    int h,
    int w,
    int64_t &orig_h,
    int64_t &orig_w
) {
    cv::Mat img = cv::imread(path);
    if (img.empty())
        throw std::runtime_error("Failed to read image: " + path);

    orig_h = img.rows;
    orig_w = img.cols;

    cv::resize(img, img, cv::Size(w, h));
    img.convertTo(img, CV_32F, 1.0 / 255);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> chw(3 * h * w);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float v = img.at<cv::Vec3f>(y, x)[c];
                chw[c * h * w + y * w + x] = (v - mean[c]) / stdv[c];
            }
        }
    }
    return chw;
}

float ObjectDetectionTask::IoU(const Box &a, const Box &b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Box> ObjectDetectionTask::nms(const std::vector<Box> &boxes) const {
    if (boxes.empty()) return {};
    std::vector<int> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(
        idx.begin(),
        idx.end(),
        [&](int i, int j) { return boxes[i].score > boxes[j].score; }
    );

    std::vector<Box> keep;
    std::vector<char> removed(boxes.size(), 0);
    for (size_t i = 0; i < idx.size(); ++i) {
        if (removed[i]) continue;
        const Box &cur = boxes[idx[i]];
        keep.push_back(cur);
        for (size_t j = i + 1; j < idx.size(); ++j) {
            if (removed[j]) continue;
            if (cur.label != boxes[idx[j]].label) continue;
            if (IoU(cur, boxes[idx[j]]) > nms_iou_thresh_)
                removed[j] = 1;
        }
    }
    return keep;
}

std::unique_ptr<ProcessedBatchData> ObjectDetectionTask::preprocess(const RawBatchData &raw) {
    const RawColumnData *col_img = nullptr;
    for (const auto &col: raw.columns)
        if (col.name == input_columns[0] && col.type == ColumnType::String) {
            col_img = &col;
            break;
        }

    if (!col_img) {
        throw std::runtime_error("Image path column not found");
    }

    size_t B = col_img->string_data.size();
    auto batch = std::make_unique<ObjectDetectionBatchData>();
    batch->batch_size = B;
    batch->img_h = img_h_;
    batch->img_w = img_w_;
    batch->pixel_values.resize(B * 3 * img_h_ * img_w_);
    batch->orig_heights.resize(B);
    batch->orig_widths.resize(B);

    for (size_t i = 0; i < B; ++i) {
        int64_t oh = 0, ow = 0;
        auto chw = preprocessImage(col_img->string_data[i], img_h_, img_w_, oh, ow);
        std::copy(chw.begin(), chw.end(),
                  batch->pixel_values.begin() + i * chw.size());
        batch->orig_heights[i] = oh;
        batch->orig_widths[i] = ow;
    }
    return batch;
}

std::vector<std::any> ObjectDetectionTask::inference(
    const ProcessedBatchData &proc,
    Ort::Session &sess,
    Ort::MemoryInfo &mem
) {
    const auto &batch = dynamic_cast<const ObjectDetectionBatchData &>(proc);
    auto inputs = batch.toOrtValues(mem);

    std::vector<const char *> inp_names, out_names = {"logits", "pred_boxes"};
    std::vector<Ort::Value> inp_vals;
    inp_names.reserve(inputs.size());
    for (auto &[n,v]: inputs) {
        inp_names.push_back(n.c_str());
        inp_vals.emplace_back(std::move(v));
    }

    auto outputs = sess.Run(
        Ort::RunOptions{nullptr},
        inp_names.data(),
        inp_vals.data(),
        inp_vals.size(),
        out_names.data(),
        2
    );

    const float *logits = outputs[0].GetTensorData<float>();
    const float *boxes = outputs[1].GetTensorData<float>();
    const auto log_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t B = log_shape[0], Q = log_shape[1], C = log_shape[2];

    std::vector<std::any> results;
    results.reserve(B);

    for (int64_t b = 0; b < B; ++b) {
        std::vector<Box> dets;

        for (int64_t q = 0; q < Q; ++q) {
            const float *row = logits + (b * Q + q) * C;
            int best_cls = std::distance(row, std::max_element(row, row + C));
            if (best_cls == C - 1) continue;
            // softmax
            float sum_e = 0.f;
            for (int k = 0; k < C; ++k) sum_e += std::exp(row[k]);
            float score = std::exp(row[best_cls]) / sum_e;
            if (score < score_thresh_) continue;

            const float *pb = boxes + (b * Q + q) * 4;
            float cx = pb[0], cy = pb[1], w = pb[2], h = pb[3];

            Box bx;
            bx.x1 = (cx - 0.5f * w) * batch.orig_widths[b];
            bx.y1 = (cy - 0.5f * h) * batch.orig_heights[b];
            bx.x2 = (cx + 0.5f * w) * batch.orig_widths[b];
            bx.y2 = (cy + 0.5f * h) * batch.orig_heights[b];
            bx.label = best_cls;
            bx.score = score;
            dets.emplace_back(bx);
        }
        results.emplace_back(std::any(nms(dets)));
    }
    return results;
}

float ObjectDetectionTask::getMetric(
    const std::vector<std::any> &predictions,
    const std::vector<std::any> &ground_truth
) const {
    size_t N = std::min(predictions.size(), ground_truth.size());
    if (N == 0) return 0.0f;

    float total_ap = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        const auto &pred_boxes = std::any_cast<const std::vector<Box> &>(predictions[i]);
        const auto &gt_boxes = std::any_cast<const std::vector<Box> &>(ground_truth[i]);
        std::vector<bool> gt_matched(gt_boxes.size(), false);
        int tp = 0, fp = 0;
        for (const auto &pred: pred_boxes) {
            bool matched = false;
            for (size_t j = 0; j < gt_boxes.size(); ++j) {
                if (gt_matched[j]) continue;
                if (pred.label != gt_boxes[j].label) continue;
                if (IoU(pred, gt_boxes[j]) >= 0.5f) {
                    matched = true;
                    gt_matched[j] = true;
                    break;
                }
            }
            if (matched)
                tp++;
            else
                fp++;
        }
        int fn = std::count(gt_matched.begin(), gt_matched.end(), false);
        float precision = tp + fp > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        float recall = tp + fn > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        float ap = (precision + recall > 0) ? (precision * recall) / (precision + recall) : 0.0f;
        total_ap += ap;
    }
    return total_ap / N;
}
