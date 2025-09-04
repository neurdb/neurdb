#include "../../hnswlib/hnswlib.h"

#include <chrono>


// Helper function: quantize a float vector.
// The quantized layout is:
// [scale (float), zero_point (uint8_t), quantized vector (dim uint8_t values)]
uint8_t* quantize_vector(const float* vec, int dim, float scale, uint8_t zero_point) {
    size_t buffer_size = sizeof(float) + sizeof(uint8_t) + dim * sizeof(uint8_t);
    uint8_t* quant_buffer = new uint8_t[buffer_size];

    memcpy(quant_buffer, &scale, sizeof(float));
    memcpy(quant_buffer + sizeof(float), &zero_point, sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
        int q = static_cast<int>(std::round(vec[i] / scale));
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        quant_buffer[sizeof(float) + sizeof(uint8_t) + i] = static_cast<uint8_t>(q);
    }
    return quant_buffer;
}

int main() {
    int dim = 16;              // Number of quantized values per vector.
    int max_elements = 100;   // For testing, a smaller number of elements.
    int M = 16;                // HNSW parameter.
    int ef_construction = 200; // HNSW parameter.

    // Create a QuantizedL2Space.
    hnswlib::QuantizedL2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Prepare random data.
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    std::vector<uint8_t*> quantized_data_vec;

    // For testing, assume the input floats are in [0,1]. A typical choice is scale = 1/255 and zero_point = 0
    float scale = 1.0f / 255.0f;
    uint8_t zero_point = 0;

    // Generate and add data to the index
    for (int i = 0; i < max_elements; i++) {
        std::vector<float> vec(dim);
        for (int j = 0; j < dim; j++) {
            vec[j] = distrib(rng);
        }
        uint8_t* qvec = quantize_vector(vec.data(), dim, scale, zero_point);
        quantized_data_vec.push_back(qvec);
        alg_hnsw->addPoint(qvec, i);
    }

    int sample_label = 1;
    try {
        std::vector<uint8_t> retrieved_vec = alg_hnsw->getDataByLabel<uint8_t>(sample_label);
        bool match = true;
        for (int j = 0; j < dim; j++) {
            if (retrieved_vec[j] != quantized_data_vec[sample_label][j]) {
                match = false;
                break;
            }
        }
        if (match) {
            std::cout << "label " << sample_label << " data matches!" << std::endl;
        } else {
            std::cout << "label " << sample_label << " data does not match!" << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    // Query: for each element, search for its nearest neighbor.
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_elements; i++) {
        // We use the same quantized vector as query.
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(quantized_data_vec[i], 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Search time: " << elapsed_seconds.count() << "s\n";

    float recall = static_cast<float>(correct) / max_elements;
    std::cout << "Recall: " << recall << "\n";

    for (auto ptr : quantized_data_vec) {
        delete[] ptr;
    }
    delete alg_hnsw;
    return 0;
}
