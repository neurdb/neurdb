#include "neurstore/neurstore.h"

#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <thread>
#include <unistd.h>

#include "neurstore/compress/compressor.h"
#include "neurstore/inference/data_loader.h"
#include "neurstore/inference/inference_task.h"
#include "neurstore/inference/sequence_classification_task.h"
#include "neurstore/inference/image_classification_task.h"
#include "neurstore/inference/tabular_classification_task.h"
#include "neurstore/inference/object_detection_task.h"
#include "neurstore/inference/tabular_regression_task.h"
#include "neurstore/utils/model.h"

/*********************************** C++ ***********************************/
NeurStore::NeurStore(std::string &storePath, const std::shared_ptr<IndexCacheManager> &indexCacheManager)
    : storePath_(storePath), indexCacheManager_(indexCacheManager) {
    if (!std::filesystem::exists(storePath_)) {
        std::filesystem::create_directories(storePath_);
    }
}

bool NeurStore::saveModel(const std::string &modelName, double tolerance, const std::string &modelPath) const {
    auto model = Model(modelPath.c_str());
    auto compressor = Compressor(tolerance, true, 8, indexCacheManager_);
    auto [modelStructure, tensorPage] = compressor.compress(model);
    auto serializedModelStructure = modelStructure.serialize();
    auto serializedTensorPage = tensorPage.serialize();
    saveModelInternal(modelName, serializedModelStructure, serializedTensorPage);
    return true;
}

bool NeurStore::saveModels(
    const std::vector<std::string> &modelNames,
    double tolerance,
    const std::string &folderPath
) const {
    Compressor compressor(tolerance, true, 8, indexCacheManager_);

    // #pragma omp parallel for schedule(dynamic)
    for (const auto &modelName: modelNames) {
        std::string modelPath = folderPath + "/" + modelName + ".onnx";
        Model model(modelPath.c_str());
        auto [modelStructure, tensorPage] = compressor.compress(model);
        auto serializedModelStructure = modelStructure.serialize();
        auto serializedTensorPage = tensorPage.serialize();
        saveModelInternal(modelName, serializedModelStructure, serializedTensorPage);
    }
    return true;
}

bool NeurStore::saveModelsFromPaths(const std::vector<std::string> &modelPaths, double tolerance) const {
    Compressor compressor(tolerance, true, 8, indexCacheManager_);

    for (const auto &modelPath: modelPaths) {
        std::filesystem::path path(modelPath);
        std::string modelName = path.stem().string();

        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Model file does not exist: " << modelPath << std::endl;
            continue;
        }

        Model model(modelPath.c_str());
        auto [modelStructure, tensorPage] = compressor.compress(model);
        auto serializedModelStructure = modelStructure.serialize();
        auto serializedTensorPage = tensorPage.serialize();
        saveModelInternal(modelName, serializedModelStructure, serializedTensorPage);
    }
    return true;
}

std::pair<std::shared_ptr<onnx::ModelProto>, double> NeurStore::saveModelDryRun(
    double tolerance,
    const std::string &modelPath,
    int load_mode
) const {
    uintmax_t originalSize = std::filesystem::file_size(modelPath);
    auto model = Model(modelPath.c_str());
    auto compressor = Compressor(tolerance, true, 8, indexCacheManager_);
    auto [modelStructure, tensorPage] = compressor.compress(model);
    auto structureSize = modelStructure.serialize().size();
    auto tensorPageSize = tensorPage.serialize().size();
    double compressionRatio = static_cast<double>(originalSize) / (structureSize + tensorPageSize);

    std::shared_ptr<onnx::ModelProto> model_proto;
    switch (load_mode) {
        case LoadMode::Float32:
            model_proto = std::make_shared<onnx::ModelProto>(
                compressor.decompress(modelStructure, tensorPage).model
            );
        break;
        case LoadMode::Int8:
            model_proto = std::make_shared<onnx::ModelProto>(
                compressor.reconstructModelUInt8(tensorPage, modelStructure)->model
            );
        break;
        case LoadMode::Int8Delta:
            model_proto = std::make_shared<onnx::ModelProto>(
                compressor.reconstructModelUInt8WithDelta(tensorPage, modelStructure)->model
            );
        break;
        case LoadMode::Float16:
            model_proto = std::make_shared<onnx::ModelProto>(
                compressor.decompressFloat16(modelStructure, tensorPage).model
            );
        break;
        default:
            std::cerr << "Unsupported load mode." << std::endl;
        return {nullptr, 0.0};
    }
    return {model_proto, compressionRatio};
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModel(const std::string &modelName) const {
    Compressor compressor = Compressor(0.0, true, 8, indexCacheManager_);
    auto model_proto = loadModel(modelName, compressor);
    return model_proto;
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModel(const std::string &modelName, Compressor &compressor) const {
    ModelStructure modelStructure;
    TensorPage tensorPage;
    loadModelMetaInternal(modelName, modelStructure, tensorPage);
    Model model = compressor.decompress(modelStructure, tensorPage);
    return std::make_shared<onnx::ModelProto>(model.model);
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsFloat16(const std::string &modelName) const {
    Compressor compressor = Compressor(0.0, true, 8, indexCacheManager_);
    auto model_proto = loadModelAsFloat16(modelName, compressor);
    return model_proto;
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsFloat16(
    const std::string &modelName,
    Compressor &compressor
) const {
    ModelStructure modelStructure;
    TensorPage tensorPage;
    loadModelMetaInternal(modelName, modelStructure, tensorPage);
    Model model = compressor.decompressFloat16(modelStructure, tensorPage);
    // Decompress the model
    auto model_proto = std::make_shared<onnx::ModelProto>(model.model);
    return model_proto;
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsInt8(const std::string &modelName) const {
    Compressor compressor = Compressor(0.0, true, 8, indexCacheManager_);
    return loadModelAsInt8(modelName, compressor);
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsInt8(
    const std::string &modelName,
    Compressor &compressor
) const {
    ModelStructure modelStructure;
    TensorPage tensorPage;
    loadModelMetaInternal(modelName, modelStructure, tensorPage);
    auto model = compressor.reconstructModelUInt8(tensorPage, modelStructure);
    auto model_proto = std::make_shared<onnx::ModelProto>(model->model);
    return model_proto;
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsInt8WithDelta(const std::string &modelName) const {
    auto compressor = Compressor(0.0, true, 8, indexCacheManager_);
    return loadModelAsInt8WithDelta(modelName, compressor);
}

std::shared_ptr<onnx::ModelProto> NeurStore::loadModelAsInt8WithDelta(
    const std::string &modelName,
    Compressor &compressor
) const {
    ModelStructure modelStructure;
    TensorPage tensorPage;
    loadModelMetaInternal(modelName, modelStructure, tensorPage);
    std::vector<uint8_t> serializedModelStructure;
    std::vector<uint8_t> serializedTensorPage;
    auto model = compressor.reconstructModelUInt8WithDelta(tensorPage, modelStructure);
    auto model_proto = std::make_shared<onnx::ModelProto>(model->model);
    return model_proto;
}

bool NeurStore::saveModelInternal(
    const std::string &modelName,
    std::vector<unsigned char> &serializedModelStructure,
    std::vector<unsigned char> &serializedTensorPage
) const {
    std::string structurePath = storePath_ + "/" + modelName + ".structure";
    std::string tensorPagePath = storePath_ + "/" + modelName + ".tensors";
    std::ofstream structure(structurePath, std::ios::binary);
    if (!structure.is_open()) {
        return false;
    }
    std::ofstream tensorPage(tensorPagePath, std::ios::binary);
    if (!tensorPage.is_open()) {
        return false;
    }
    structure.write(
        reinterpret_cast<const char *>(serializedModelStructure.data()),
        static_cast<std::streamsize>(serializedModelStructure.size())
    );
    if (!structure.good()) {
        return false;
    }
    structure.close();
    tensorPage.write(
        reinterpret_cast<const char *>(serializedTensorPage.data()),
        static_cast<std::streamsize>(serializedTensorPage.size())
    );
    if (!tensorPage.good()) {
        return false;
    }
    tensorPage.close();
    return true;
}

void NeurStore::loadModelMetaInternal(
    const std::string &modelName,
    ModelStructure &modelStructure,
    TensorPage &tensorPage
) const {
    auto mmap_file = [](const std::string &path, std::vector<uint8_t> &out) -> bool {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "NeurStore::loadModelMetaInternal: Failed to open " << path << std::endl;
            return false;
        }

        struct stat st{};
        if (fstat(fd, &st) == -1) {
            std::cerr << "NeurStore::loadModelMetaInternal: Failed to stat " << path << std::endl;
            close(fd);
            return false;
        }

        size_t size = st.st_size;
        void *mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            std::cerr << "NeurStore::loadModelMetaInternal: Failed to mmap " << path << std::endl;
            close(fd);
            return false;
        }
        out.assign(static_cast<uint8_t *>(mapped), static_cast<uint8_t *>(mapped) + size);

        munmap(mapped, size);
        close(fd);
        return true;
    };

    auto read_file = [](const std::string &path, std::vector<uint8_t> &out) -> bool {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "NeurStore::loadModelMetaInternal: open " << path << " fail\n";
            return false;
        }

        struct stat st{};
        if (fstat(fd, &st) == -1) {
            std::cerr << "NeurStore::loadModelMetaInternal: stat " << path << " fail\n";
            close(fd);
            return false;
        }

        size_t size = st.st_size;
        out.resize(size);
        ssize_t n = read(fd, out.data(), size);
        close(fd);
        return true;
    };

    std::string structurePath = storePath_ + "/" + modelName + ".structure";
    std::string tensorPagePath = storePath_ + "/" + modelName + ".tensors";

    std::vector<uint8_t> serializedModelStructure;
    std::vector<uint8_t> serializedTensorPage;

    // if (!mmap_file(structurePath, serializedModelStructure)) return;
    // if (!mmap_file(tensorPagePath, serializedTensorPage)) return;
    if (!read_file(structurePath, serializedModelStructure)) return;
    if (!read_file(tensorPagePath, serializedTensorPage)) return;

    modelStructure = ModelStructure::deserialize(serializedModelStructure);
    tensorPage = TensorPage::deserialize(serializedTensorPage);
}

std::pair<double, double> NeurStore::evaluateTolerance(
    double tolerance,
    const std::string &modelPath,
    int load_mode,
    const InferenceOptions &options
) const {
    // compressed performance
    auto [model, compressionRatio] = saveModelDryRun(
        tolerance,
        modelPath,
        load_mode
    );
    std::vector<std::any> compressed_out;
    inference(model, options, compressed_out);

    // original performance
    std::vector<std::any> original_out;
    Model originalModel(modelPath.c_str());
    std::shared_ptr<onnx::ModelProto> originalModelProto = std::make_shared<onnx::ModelProto>(originalModel.model);
    inference(originalModelProto, options, original_out);

    std::unique_ptr<InferenceTask> task;
    switch (options.task) {
        case Task::SequenceClassification:
            task = std::make_unique<SeqClassificationTask>(options.input_columns, options.output_column, options.batch_size, nullptr, 0); break;
        case Task::ImageClassification:
            task = std::make_unique<ImageClassificationTask>(options.input_columns, nullptr, 0); break;
        case Task::TabularClassification:
            task = std::make_unique<TabularClassificationTask>(options.input_columns, options.output_column, 0); break;
        case Task::TabularRegression:
            task = std::make_unique<TabularRegressionTask>(options.input_columns, options.output_column, 0); break;
        case Task::ObjectDetection:
            task = std::make_unique<ObjectDetectionTask>(options.input_columns, options.output_column, 0); break;
        default:
            throw std::runtime_error("Unsupported task type for metric evaluation");
    }
    double performance_delta = task->getMetric(original_out, compressed_out);
    return {compressionRatio, performance_delta};
}

void NeurStore::cleanCache() const {
    indexCacheManager_->clearCache();
}

uintmax_t NeurStore::totalSize() const {
    if (!fs::exists(storePath_) || !fs::is_directory(storePath_)) {
        return 0;
    }

    uintmax_t totalSize = 0;
    for (const auto &entry: fs::directory_iterator(storePath_)) {
        if (fs::is_regular_file(entry)) {
            totalSize += fs::file_size(entry) / 1024 / 1024;
        }
    }
    return totalSize;
}

uintmax_t NeurStore::modelSize() const {
    if (!fs::exists(storePath_) || !fs::is_directory(storePath_)) {
        return 0;
    }

    uintmax_t modelSize = 0;
    for (const auto &entry: fs::directory_iterator(storePath_)) {
        if (
            fs::is_regular_file(entry) &&
            (entry.path().extension() == ".structure"
             || entry.path().extension() == ".tensors")
        ) {
            modelSize += fs::file_size(entry.path()) / 1024 / 1024;
        }
    }
    return modelSize;
}

uintmax_t NeurStore::indexSize() const {
    if (!fs::exists(storePath_) || !fs::is_directory(storePath_)) {
        return 0;
    }

    uintmax_t totalSize = 0;
    for (const auto &entry: fs::directory_iterator(storePath_)) {
        if (fs::is_regular_file(entry) && entry.path().extension() == ".index") {
            totalSize += fs::file_size(entry.path()) / 1024 / 1024;
        }
    }
    return totalSize;
}

uint32_t NeurStore::modelCount() const {
    if (!fs::exists(storePath_) || !fs::is_directory(storePath_)) {
        return 0;
    }

    uint32_t count = 0;
    for (const auto &entry: fs::directory_iterator(storePath_)) {
        if (fs::is_regular_file(entry)) {
            if (entry.path().extension() == ".structure") {
                count++;
            }
        }
    }
    return count;
}

uint32_t NeurStore::indexCount() const {
    if (!fs::exists(storePath_) || !fs::is_directory(storePath_)) {
        return 0;
    }

    uint32_t count = 0;
    for (const auto &entry: fs::directory_iterator(storePath_)) {
        if (fs::is_regular_file(entry)) {
            if (entry.path().extension() == ".index") {
                count++;
            }
        }
    }
    return count;
}

void NeurStore::inference(
    const std::shared_ptr<onnx::ModelProto> &model,
    const InferenceOptions &options,
    std::vector<std::any> &out
) const {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");
    Ort::SessionOptions sess_opts;
    sess_opts.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sess_opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    if (options.use_gpu) {
        OrtCUDAProviderOptions cuda_opts{};
        sess_opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

    std::string buffer;
    model->SerializeToString(&buffer);

    Ort::Session session(env, buffer.data(), buffer.size(), sess_opts);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    if (!options.tokenizer_path.empty()) {
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(options.tokenizer_path));
    }

    std::unique_ptr<InferenceTask> task;
    switch (options.task) {
        case Task::SequenceClassification:
            task = std::make_unique<SeqClassificationTask>(
                options.input_columns
                , options.output_column,
                options.batch_size,
                std::move(tokenizer),
                options.max_input_len
            );
            break;
        case Task::ImageClassification:
            task = std::make_unique<ImageClassificationTask>(
                options.input_columns,
                options.output_column,
                options.batch_size
            );
            break;
        case Task::TabularClassification:
            task = std::make_unique<TabularClassificationTask>(
                options.input_columns,
                options.output_column,
                options.batch_size
            );
            break;
        case Task::ObjectDetection:
            task = std::make_unique<ObjectDetectionTask>(
                options.input_columns,
                options.output_column,
                options.batch_size,
                400,
                400,
                0.1f,
                0.5f
            );
            break;
        case Task::TabularRegression:
            task = std::make_unique<TabularRegressionTask>(
                options.input_columns,
                options.output_column,
                options.batch_size
            );
            break;
        default:
            std::cerr << "Unsupported task type." << std::endl;
            return;
    }

    PostgresBatchReader reader = PostgresBatchReader(options);
    out.clear();
    while (reader.hasNext()) {
        RawBatchData raw = reader.next();
        auto processed = task->preprocess(raw);
        auto results = task->inference(*processed, session, memory_info);
        out.insert(out.end(), results.begin(), results.end());
    }
    reader.close();
}

void NeurStore::inference(const InferenceOptions &options, std::vector<std::any>& out) const {
    std::shared_ptr<onnx::ModelProto> model;
    switch (options.load_mode) {
        case LoadMode::Float32:
            model = loadModel(options.model_id);
            break;
        case LoadMode::Int8:
            model = loadModelAsInt8(options.model_id);
            break;
        case LoadMode::Int8Delta:
            model = loadModelAsInt8WithDelta(options.model_id);
            break;
        case LoadMode::Float16:
            model = loadModelAsFloat16(options.model_id);
            break;
        default:
            std::cerr << "Unsupported load mode." << std::endl;
    }
    inference(model, options, out);
}

std::string NeurStore::summary() const {
    std::ostringstream oss;
    uintmax_t totalSize = this->totalSize();
    uintmax_t modelSize = this->modelSize();
    uintmax_t indexSize = this->indexSize();
    oss << "=========================================" << std::endl;
    oss << "            NeurStore Summary            " << std::endl;
    oss << "=========================================" << std::endl;
    oss << "Store Path: " << storePath_ << std::endl;
    oss << "Model Count: " << modelCount() << std::endl;
    oss << "Index Count: " << indexCount() << std::endl;
    oss << "=========================================" << std::endl;
    oss << "Total Size: " << totalSize << " MB" << std::endl;
    oss << "Model Size: " << modelSize << " MB" << " (" << static_cast<double>(modelSize) /
            static_cast<double>(totalSize) * 100 << "%)" << std::endl;
    oss << "Index Size: " << indexSize << " MB" << " (" << static_cast<double>(indexSize) /
            static_cast<double>(totalSize) * 100 << "%)" << std::endl;
    oss << "=========================================" << std::endl;
    return oss.str();
}

/************************************ C ************************************/
bool set_parallelism(int parallelism) {
    if (parallelism <= 0) {
        return false;
    }
    omp_set_num_threads(parallelism);
    return true;
}

int get_parallelism() {
    return omp_get_max_threads();
}

NeurStoreC *ns_create(const char *store_path, IndexCacheManagerC *cache_mgr) {
    std::string path(store_path);
    std::shared_ptr<IndexCacheManager> sp(
        reinterpret_cast<IndexCacheManager *>(cache_mgr),
        [](IndexCacheManager *) {
        }
    );
    return reinterpret_cast<NeurStoreC *>(
        new NeurStore(path, sp)
    );
}

void ns_destroy(NeurStoreC *nstore) {
    delete reinterpret_cast<NeurStore *>(nstore);
}

bool ns_save_model_internal(
    NeurStoreC *ns,
    const char *model_name,
    double tolerance,
    const char *model_path
) {
    auto store = reinterpret_cast<NeurStore *>(ns);
    return store->saveModel(model_name, tolerance, model_path);
}

bool ns_save_models_internal(
    NeurStoreC *ns,
    const char **model_names,
    int model_count,
    double tolerance,
    const char *folder_path
) {
    auto store = reinterpret_cast<NeurStore *>(ns);
    std::vector<std::string> names(model_names, model_names + model_count);
    return store->saveModels(names, tolerance, folder_path);
}

bool ns_save_model_dry_run_internal(
    NeurStoreC* ns,
    double tolerance,
    const char* model_path,
    int load_mode,
    InferenceOptionsC *options,
    double* ratio_out,
    double* delta_performance_out
) {
    try {
        auto store = reinterpret_cast<NeurStore*>(ns);
        InferenceOptions opts(*options);
        auto res = store->evaluateTolerance(
            tolerance,
            model_path,
            load_mode,
            opts
        );
        *ratio_out = res.first;
        *delta_performance_out = res.second;
        return true;
    } catch (const std::exception &e) {
        std::cout << "ns_save_model_dry_run_internal: " << e.what() << std::endl;
        return false;
    }
}

ModelC *ns_load_model_internal(NeurStoreC *ns, const char *model_name) {
    auto model_ptr = ns->loadModel(model_name);
    if (!model_ptr) {
        return nullptr;
    }
    return new Model(std::move(*model_ptr));
}

ModelC *ns_load_model_intermal_uint8(NeurStoreC *ns, const char *model_name) {
    auto model_ptr = ns->loadModelAsInt8(model_name);
    if (!model_ptr) {
        return nullptr;
    }
    return new Model(*model_ptr);
}

ModelC *ns_load_model_intermal_uint8_delta(NeurStoreC *ns, const char *model_name) {
    auto model_ptr = ns->loadModelAsInt8WithDelta(model_name);
    if (!model_ptr) {
        return nullptr;
    }
    return new Model(*model_ptr);
}

ModelC *ns_load_model_intermal_float16(NeurStoreC *ns, const char *model_name) {
    auto model_ptr = ns->loadModelAsFloat16(model_name);
    if (!model_ptr) {
        return nullptr;
    }
    return new Model(*model_ptr);
}

void ns_free_model(ModelC *model) {
    delete reinterpret_cast<Model *>(model);
}

int ns_inference_internal(
    NeurStoreC *ns,
    InferenceOptionsC *options,
    const char **err_msg_out
) {
    try {
        auto store = reinterpret_cast<NeurStore *>(ns);
        InferenceOptions opts(*options);
        std::vector<std::any> out;
        store->inference(opts, out);
        return 1;
    } catch (const std::exception &e) {
        if (err_msg_out)
            *err_msg_out = strdup(e.what());
        return 0;
    }
}
