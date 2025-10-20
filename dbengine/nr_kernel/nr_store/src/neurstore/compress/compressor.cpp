#include "neurstore/compress/compressor.h"

#include <omp.h>


#include "neurstore/compress/index/router/single_index_router.h"
#include "neurstore/compress/method/linear_quantization.h"
#include "neurstore/compress/strategy/standard_nbits_strategy.h"
#include "neurstore/compress/utils/matrix_utils.h"
#include "neurstore/utils/global.h"

/*********************************** C++ ***********************************/

Compressor::Compressor(
    double tolerance,
    bool dynamic,
    int default_quantized_bit_width,
    std::shared_ptr<IndexCacheManager> index_cache_manager
): tolerance_(tolerance),
   dynamic_(dynamic),
   default_quantized_bit_width_(default_quantized_bit_width),
   delta_quant_compress_(tolerance, dynamic, default_quantized_bit_width),
   selection_strategy_(std::make_shared<StandardNBitsStrategy>(tolerance, default_quantized_bit_width)),
   index_cache_manager_(std::move(index_cache_manager)) {
    if (!dynamic && default_quantized_bit_width <= 0) {
        throw std::invalid_argument("Compressor: default quantized bit width must be positive if dynamic is false");
    }
}

std::pair<ModelStructure, TensorPage> Compressor::compress(
    const Model &model
) {
    auto model_weight = extractModelTensors(model);
    const_cast<onnx::ModelProto&>(model.model).mutable_graph()->clear_initializer();
    std::vector<uint8_t> serialized_model_structure;
    serialized_model_structure.resize(model.model.ByteSizeLong());
    model.model.SerializeToArray(
        serialized_model_structure.data(),
        static_cast<int>(serialized_model_structure.size())
    );
    ModelStructure model_structure(serialized_model_structure, {});

    std::vector<std::shared_ptr<TensorPacket> > tensor_packets(model_weight->nTensors());
    std::vector<std::string> tensor_names(model_weight->nTensors());
    std::vector<std::unordered_map<int, std::vector<PendingInsertion>>> pending_insertions_map(omp_get_max_threads());

    #pragma omp parallel for schedule(static, 8)
    for (uint i = 0; i < model_weight->nTensors(); i++) {
        int thread_id = omp_get_thread_num();
        std::shared_ptr<TensorF64> tensor = std::move(model_weight->tensors[i]);
        tensor_names[i] = std::move(model_weight->names[i]);
        if (model_weight->isIntType(i)) {
            // If the tensor is an integer type, we skip compression
            // and directly create an IntTensorPacket.
            auto int_type = model_weight->getIntType(i);
            auto int_tensor_packet = std::make_shared<IntTensorPacket>(
                int_type,
                tensor
            );
            tensor_packets[i] = int_tensor_packet;
        } else {
            auto packet = compressTensor(tensor, i, pending_insertions_map[thread_id]);
            tensor_packets[i] = packet;
        }
    }

    std::unordered_map<std::pair<int, int>, std::vector<PendingInsertion>, pair_hash> merged_pending_insertions_map;

    for (auto& local_map : pending_insertions_map) {
        for (auto& [index_id, pending_list] : local_map) {
            for (auto& pending : pending_list) {
                auto key = std::make_pair(pending.dimension, index_id);
                auto& merged_list = merged_pending_insertions_map[key];
                merged_list.emplace_back(std::move(pending));
            }
        }
    }

    std::vector<decltype(merged_pending_insertions_map)::const_iterator> index_iterator_list;
    index_iterator_list.reserve(merged_pending_insertions_map.size());
    for (auto iterator = merged_pending_insertions_map.begin(); iterator != merged_pending_insertions_map.end(); ++iterator) {
        index_iterator_list.emplace_back(iterator);
    }

    // #pragma omp parallel for schedule(static, 8)
    for (size_t i = 0; i < index_iterator_list.size(); ++i) {
        const auto& iterator = index_iterator_list[i];
        const auto& [dimension, index_id] = iterator->first;
        const std::vector<PendingInsertion>& pending_list = iterator->second;

        auto index = index_cache_manager_->get(dimension, index_id);
        for (const auto& pi : pending_list) {
            auto rep_id = index->insert(pi.tensor, pi.scale, pi.zero_point);
            tensor_packets[pi.tensor_sequence_num]->setBaseTensorId(rep_id);
        }
        index_cache_manager_->release(index, dimension, index_id, true);
    }

    auto tensor_page = TensorPage(
        tensor_packets, tensor_names
    );

    return std::make_pair(model_structure, tensor_page);
}

Model Compressor::decompress(
    const ModelStructure &model_structure,
    const TensorPage &tensor_page
) {
    const size_t n_tensors = tensor_page.nTensors();
    std::vector<std::shared_ptr<TensorF64>> tensors(n_tensors);
    std::vector<std::string> names(n_tensors);
    std::vector<std::optional<IntTensorPacket::IntType>> int_types(n_tensors);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        auto tensor_packet = tensor_page.getTensorPacket(i);
        if (auto int_packet = std::dynamic_pointer_cast<IntTensorPacket>(tensor_packet)) {
            // If the tensor is an integer type, we directly create a TensorF64 from IntTensorPacket.
            tensors[i] = std::make_shared<TensorF64>(
                int_packet->toTensor64()
            );
            int_types[i] = int_packet->getIntType();
            names[i] = tensor_page.getName(i);
            continue;
        }

        int64_t base_tensor_id = tensor_packet->getBaseTensorId();
        if (base_tensor_id == -1) {
            throw std::invalid_argument("Compressor::decompress: invalid base tensor id.");
        }
        auto index = index_cache_manager_->get(
            tensor_packet->getDimension(),
            tensor_packet->getIndexId()
        );
        auto base_tensor = index->retrieve(base_tensor_id);
        index_cache_manager_->release(
            index, tensor_packet->getDimension(), tensor_packet->getIndexId(), false
        );

        std::shared_ptr<TensorF64> decompressed_tensor = delta_quant_compress_.decompress(
            base_tensor,
            tensor_packet
        );
        tensors[i] = decompressed_tensor;
        names[i] = tensor_page.getName(i);
    }
    ModelWeight model_weight(tensors, names, int_types);
    std::shared_ptr<Model> model = reconstructModel(model_weight, model_structure);
    return *model;
}

Model Compressor::decompressFloat16(
    const ModelStructure &model_structure,
    const TensorPage &tensor_page
) {
    const size_t n_tensors = tensor_page.nTensors();
    std::vector<std::shared_ptr<TensorF16>> tensors(n_tensors);
    std::vector<std::string> names(n_tensors);
    std::vector<std::optional<IntTensorPacket::IntType>> int_types(n_tensors);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        auto tensor_packet = tensor_page.getTensorPacket(i);
        if (auto int_packet = std::dynamic_pointer_cast<IntTensorPacket>(tensor_packet)) {
            // If the tensor is an integer type, we directly create a TensorF16 from IntTensorPacket.
            tensors[i] = std::make_shared<TensorF16>(
                int_packet->toTensor16()
            );
            int_types[i] = int_packet->getIntType();
            names[i] = tensor_page.getName(i);
            continue;
        }

        int64_t base_tensor_id = tensor_packet->getBaseTensorId();
        if (base_tensor_id == -1) {
            throw std::invalid_argument("Compressor::decompress: invalid base tensor id.");
        }
        auto index = index_cache_manager_->get(tensor_packet->getDimension(), tensor_packet->getIndexId());
        auto base_tensor_f16 = index->retrieveF16(base_tensor_id);
        index_cache_manager_->release(index, tensor_packet->getDimension(), tensor_packet->getIndexId(), false);

        std::shared_ptr<TensorF16> decompressed_tensor = delta_quant_compress_.decompressF16(
            base_tensor_f16,
            tensor_packet
        );
        tensors[i] = decompressed_tensor;
        names[i] = tensor_page.getName(i);
    }

    ModelWeightF16 model_weight_f16(tensors, names, int_types);
    std::shared_ptr<Model> model = reconstructModelF16(model_weight_f16, model_structure);
    return *model;
}

std::shared_ptr<TensorPacket> Compressor::compressTensor(
    const std::shared_ptr<TensorF64> &tensor,
    const uint tensor_index,
    std::unordered_map<
        int, std::vector<PendingInsertion>
    > &pending_insertion_tensors
) {
    // 8-bit quant
    TensorType::VectorFloat64 tensor_data = tensor->getTensor();
    TensorType::VectorUInt8 quantized_tensor;
    double scale;
    double zero_point;
    LinearQuantization::linearAsymmetricQuantize(
        tensor_data,
        8,
        quantized_tensor,
        scale,
        zero_point
    );
    std::shared_ptr<VectorFloat64Router> router = index_cache_manager_->getRouter(tensor->getDimension());
    int index_id = router->route(tensor_data);
    auto index = index_cache_manager_->get(tensor->getDimension(), index_id);

    auto representative_id = index->query(quantized_tensor, scale, zero_point);

    // if the dimension of the tensor is 1, we directly insert it into the index
    // TODO: this is a special case for 1D tensors, we should fix this corner case in the future
    if (tensor->getDimension() == 1) {
        representative_id = -1;
    }

    TensorType::VectorFloat64 base_tensor;
    if (representative_id == -1) {
        TensorType::VectorFloat64 tensor_dequantized;
        LinearQuantization::linearAsymmetricDequantize(
            quantized_tensor,
            scale,
            zero_point,
            tensor_dequantized
        );
        int dimension = tensor->getDimension();
        pending_insertion_tensors[index_id].emplace_back(
            dimension,
            index_id,
            std::move(quantized_tensor),
            scale,
            zero_point,
            tensor_index
        );
        base_tensor = tensor_dequantized;
        router->update(
            index_id, tensor_data, static_cast<int>(index->size())
        );
    } else {
        base_tensor = index->retrieve(representative_id);
        if (selection_strategy_->selectAsRepresentative(base_tensor, tensor_data)) {
            TensorType::VectorFloat64 tensor_dequantized;
            LinearQuantization::linearAsymmetricDequantize(
                quantized_tensor,
                scale,
                zero_point,
                tensor_dequantized
            );
            int dimension = tensor->getDimension();
            pending_insertion_tensors[index_id].emplace_back(
                dimension,
                index_id,
                std::move(quantized_tensor),
                scale,
                zero_point,
                tensor_index
            );
            base_tensor = tensor_dequantized;
            representative_id = -1;
            router->update(
                index_id, tensor_data, static_cast<int>(index->size())
            );
        }
    }
    index_cache_manager_->release(index, tensor->getDimension(), index_id, false);

    std::shared_ptr<TensorPacket> packet = delta_quant_compress_.compress(
        base_tensor,
        tensor_data,
        tensor->getShape()
    );
    packet->setBaseTensorId(representative_id);
    packet->setIndexId(index_id);
    return packet;
}

std::shared_ptr<TensorF64> Compressor::decompressTensor(
    const std::shared_ptr<TensorPacket> &tensor_packet,
    const std::shared_ptr<VectorUInt8Index> &index
) {
    int64_t base_tensor_id = tensor_packet->getBaseTensorId();
    if (base_tensor_id == -1) {
        throw std::invalid_argument("Compressor::decompressTensor: invalid base tensor id.");
    }
    auto base_tensor = index->retrieve(base_tensor_id);
    return delta_quant_compress_.decompress(base_tensor, tensor_packet);
}

std::shared_ptr<ModelWeight> Compressor::extractModelTensors(const Model &model) {
    auto model_weight = std::make_shared<ModelWeight>(
        std::vector<std::shared_ptr<TensorF64> >(),
        std::vector<std::string>(),
        std::vector<std::optional<IntTensorPacket::IntType>>()
    );
    const auto &graph = model.model.graph();
    const size_t num_tensors = graph.initializer_size();

    std::vector<std::string> names(num_tensors);
    std::vector<std::shared_ptr<TensorF64> > tensors(num_tensors);
    std::vector<std::optional<IntTensorPacket::IntType>> int_types(num_tensors, std::nullopt);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(num_tensors); ++i) {
        const auto &initializer = graph.initializer(i);
        int dtype = initializer.data_type();
        if (dtype != onnx::TensorProto::FLOAT &&
            dtype != onnx::TensorProto::FLOAT16 &&
            dtype != onnx::TensorProto::DOUBLE) {
            std::cout << "Got INT type weight at index: " << i << std::endl;
            if (dtype == onnx::TensorProto::INT8) {
                int_types[i] = IntTensorPacket::IntType::INT8;
            } else if (dtype == onnx::TensorProto::INT16) {
                int_types[i] = IntTensorPacket::IntType::INT16;
            } else if (dtype == onnx::TensorProto::INT32) {
                int_types[i] = IntTensorPacket::IntType::INT32;
            } else if (dtype == onnx::TensorProto::UINT8) {
                int_types[i] = IntTensorPacket::IntType::UINT8;
            } else if (dtype == onnx::TensorProto::UINT16) {
                int_types[i] = IntTensorPacket::IntType::UINT16;
            } else if (dtype == onnx::TensorProto::UINT32) {
                int_types[i] = IntTensorPacket::IntType::UINT32;
            } else {
                // Unsupported data type, skip this tensor
                throw std::invalid_argument(
                    "Compressor::extractModelTensors: Unsupported data type in initializer: " +
                    std::to_string(dtype)
                );
            }
        }
        auto tensor = std::make_shared<TensorF64>(
            MatrixUtils::tensorProto2TensorF64(initializer)
        );
        names[i] = initializer.name();
        tensors[i] = tensor;
    }
    model_weight->set(names, tensors, int_types);
    return model_weight;
}

void fillRawDataFromIntType(
    const std::vector<double>& src,
    IntTensorPacket::IntType int_type,
    onnx::TensorProto& initializer
) {
    const int dimension = static_cast<int>(src.size());
    const size_t element_size = IntTensorPacket::getIntTypeInBytes(int_type);
    std::string raw_data(dimension * element_size, 0);
    char* raw_ptr = raw_data.data();

    switch (int_type) {
        case IntTensorPacket::INT8:
            initializer.set_data_type(onnx::TensorProto::INT8);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<int8_t*>(raw_ptr)[d] = static_cast<int8_t>(src[d]);
            break;
        case IntTensorPacket::INT16:
            initializer.set_data_type(onnx::TensorProto::INT16);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<int16_t*>(raw_ptr)[d] = static_cast<int16_t>(src[d]);
            break;
        case IntTensorPacket::INT32:
            initializer.set_data_type(onnx::TensorProto::INT32);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<int32_t*>(raw_ptr)[d] = static_cast<int32_t>(src[d]);
            break;
        case IntTensorPacket::UINT8:
            initializer.set_data_type(onnx::TensorProto::UINT8);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<uint8_t*>(raw_ptr)[d] = static_cast<uint8_t>(src[d]);
            break;
        case IntTensorPacket::UINT16:
            initializer.set_data_type(onnx::TensorProto::UINT16);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<uint16_t*>(raw_ptr)[d] = static_cast<uint16_t>(src[d]);
            break;
        case IntTensorPacket::UINT32:
            initializer.set_data_type(onnx::TensorProto::UINT32);
            for (int d = 0; d < dimension; ++d)
                reinterpret_cast<uint32_t*>(raw_ptr)[d] = static_cast<uint32_t>(src[d]);
            break;
        default:
            throw std::invalid_argument("fillRawDataFromIntType: Unsupported IntType");
    }

    initializer.set_raw_data(raw_data);
}

void fillRawDataFromIntType(
    const double* src,
    int dimension,
    IntTensorPacket::IntType int_type,
    onnx::TensorProto& initializer
) {
    std::vector<double> buffer(src, src + dimension);
    fillRawDataFromIntType(buffer, int_type, initializer);
}

void fillRawDataFromIntType(
    const Eigen::half* src,
    int dimension,
    IntTensorPacket::IntType int_type,
    onnx::TensorProto& initializer
) {
    std::vector<double> buffer(dimension);
    for (int i = 0; i < dimension; ++i) {
        buffer[i] = static_cast<double>(src[i]);
    }
    fillRawDataFromIntType(buffer, int_type, initializer);
}

std::shared_ptr<Model> Compressor::reconstructModel(
    const ModelWeight &model_weight,
    const ModelStructure &model_structure
) {
    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromArray(
        model_structure.serialized_model_structure.data(),
        static_cast<int>(model_structure.serialized_model_structure.size())
    )) {
        throw std::runtime_error("Compressor::reconstructModel: Failed to parse model");
    }
    auto model = std::make_shared<Model>(onnx_model);

    auto *graph = model->model.mutable_graph();
    graph->clear_initializer();

    const size_t n_tensors = model_weight.nTensors();
    std::vector<onnx::TensorProto> tensor_protos(n_tensors);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        auto tensor = model_weight.getTensorF64(i);
        auto tensor_name = model_weight.getName(i);

        onnx::TensorProto local_initializer;
        local_initializer.set_name(tensor_name);

        for (const auto &dim : tensor->getShape()) {
            local_initializer.add_dims(dim);
        }

        if (model_weight.isIntType(i)) {
            auto int_type = model_weight.getIntType(i);
            fillRawDataFromIntType(
                tensor->getTensor().data(),
                tensor->getDimension(),
                int_type,
                local_initializer
            );
            tensor_protos[i] = std::move(local_initializer);
            continue;
        }
        local_initializer.set_data_type(onnx::TensorProto::FLOAT);
        const int dimension = tensor->getDimension();
        std::vector<float> float_buffer(dimension);
        const double* src = tensor->getTensor().data();
        for (int d = 0; d < dimension; d++) {
            float_buffer[d] = static_cast<float>(src[d]);
        }
        local_initializer.set_raw_data(
            std::string(
                reinterpret_cast<const char*>(float_buffer.data()),
                sizeof(float) * float_buffer.size()
            )
        );
        tensor_protos[i] = std::move(local_initializer);
    }
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        auto *initializer = graph->add_initializer();
        initializer->CopyFrom(tensor_protos[i]);
    }
    return model;
}

std::shared_ptr<Model> Compressor::reconstructModelF16(
    const ModelWeightF16 &model_weight,
    const ModelStructure &model_structure
) {
    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromArray(
        model_structure.serialized_model_structure.data(),
        static_cast<int>(model_structure.serialized_model_structure.size())
    )) {
        throw std::runtime_error("Compressor::reconstructModelF16: Failed to parse model");
    }
    auto model = std::make_shared<Model>(onnx_model);

    auto *graph = model->model.mutable_graph();
    graph->clear_initializer();

    const size_t n_tensors = model_weight.nTensors();
    std::vector<onnx::TensorProto> tensor_protos(n_tensors);
    std::vector<onnx::NodeProto> node_protos(n_tensors);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        auto tensor_f16 = model_weight.getTensorF16(i);
        auto tensor_name = model_weight.getName(i);

        onnx::TensorProto local_initializer;
        for (const auto &dim : tensor_f16->getShape()) {
            local_initializer.add_dims(dim);
        }
        if (model_weight.isIntType(i)) {
            local_initializer.set_name(tensor_name);
            fillRawDataFromIntType(
                tensor_f16->getTensor().data(),
                tensor_f16->getDimension(),
                model_weight.getIntType(i),
                local_initializer
            );
            tensor_protos[i] = std::move(local_initializer);
            continue;
        }
        local_initializer.set_name(tensor_name + "_f16");
        local_initializer.set_data_type(onnx::TensorProto::FLOAT16);

        const int dimension = tensor_f16->getDimension();
        const Eigen::half* src = tensor_f16->getTensor().data();

        local_initializer.set_raw_data(
            std::string(
                reinterpret_cast<const char*>(src),
                sizeof(Eigen::half) * dimension
            )
        );
        tensor_protos[i] = std::move(local_initializer);

        onnx::NodeProto node;
        node.set_op_type("Cast");
        node.add_input(tensor_name + "_f16");
        node.add_output(tensor_name);
        auto* attr = node.add_attribute();
        attr->set_name("to");
        attr->set_type(onnx::AttributeProto::INT);
        attr->set_i(1);  // 1 == float (per ONNX TensorProto.DataType)
        node_protos[i] = std::move(node);
    }

    for (const auto& node : node_protos) {
        auto* new_node = graph->add_node();
        new_node->CopyFrom(node);
    }

    auto* initializer_list = graph->mutable_initializer();
    initializer_list->Reserve(static_cast<int>(tensor_protos.size()));
    for (const auto& initializer : tensor_protos) {
        if (initializer.name().empty()) {
            continue;
        }
        auto* init = initializer_list->Add();
        init->CopyFrom(initializer);
    }

    return model;
}

std::shared_ptr<Model> Compressor::reconstructModelUInt8(
    const TensorPage &model_weight,
    const ModelStructure &model_structure
) const {
    // Parse ONNX model structure
    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromArray(
        model_structure.serialized_model_structure.data(),
        static_cast<int>(model_structure.serialized_model_structure.size())
    )) {
        throw std::runtime_error("Compressor::reconstructModelUInt8: Failed to parse model");
    }
    auto model = std::make_shared<Model>(onnx_model);

    auto *graph = model->model.mutable_graph();
    graph->clear_initializer();

    const size_t n_tensors = model_weight.names.size();
    std::vector<onnx::TensorProto> tensor_protos(n_tensors * 4);

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        const auto &tensor_name = model_weight.names[i];
        const auto &packet_ptr = model_weight.tensor_packets[i];

        if (auto int_packet = std::dynamic_pointer_cast<IntTensorPacket>(packet_ptr)) {
            const auto& tensor = int_packet->toTensor64();
            onnx::TensorProto initializer;
            initializer.set_name(tensor_name);
            for (const auto &dim : tensor.getShape()) {
                initializer.add_dims(dim);
            }
            fillRawDataFromIntType(
                tensor.getTensor().data(),
                tensor.getDimension(),
                int_packet->getIntType(),
                initializer
            );
            tensor_protos[i * 4 + 0] = std::move(initializer);
            continue;
        }

        const auto &packet = *packet_ptr;
        const int64_t base_id = packet.getBaseTensorId();
        const int dimension = packet.getDimension();
        const int index_id = packet.getIndexId();

        auto index = index_cache_manager_->get(dimension, index_id);
        if (!index) {
            throw std::runtime_error("Compressor::reconstructModelUInt8: Cannot get index for dimension " + std::to_string(dimension));
        }

        auto uint8QuantizedPacket = index->retrieveUINT8Quantized(base_id);
        index_cache_manager_->release(index, dimension, index_id, false);

        auto &raw_data = uint8QuantizedPacket.data;
        auto scale = static_cast<float>(uint8QuantizedPacket.scale);
        auto zero_point = static_cast<float>(uint8QuantizedPacket.zero_point);

        // UINT8 base tensor
        onnx::TensorProto base_initializer;
        base_initializer.set_name(tensor_name + "_base");

        base_initializer.set_data_type(onnx::TensorProto::UINT8);
        for (const auto &dim : packet.getShape()) {
            base_initializer.add_dims(dim);
        }
        base_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(raw_data.data()), raw_data.size())
        );
        tensor_protos[i * 4 + 0] = std::move(base_initializer);

        // FLOAT scale tensor
        onnx::TensorProto scale_initializer;
        scale_initializer.set_name(tensor_name + "_base_scale");
        scale_initializer.set_data_type(onnx::TensorProto::FLOAT);
        scale_initializer.add_dims(1);
        scale_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&scale), sizeof(float))
        );
        tensor_protos[i * 4 + 1] = std::move(scale_initializer);

        // INT8 zero_point tensor
        onnx::TensorProto zero_point_initializer;
        zero_point_initializer.set_name(tensor_name + "_base_zero_point");
        zero_point_initializer.set_data_type(onnx::TensorProto::UINT8);
        zero_point_initializer.add_dims(1);
        uint8_t zero_val = 0;
        zero_point_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&zero_val), sizeof(uint8_t))
        );
        tensor_protos[i * 4 + 2] = std::move(zero_point_initializer);

        // OFFSET
        float offset = - zero_point * scale;
        onnx::TensorProto offset_initializer;
        offset_initializer.set_name(tensor_name + "_base_offset");
        offset_initializer.set_data_type(onnx::TensorProto::FLOAT);
        offset_initializer.add_dims(1);
        offset_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&offset), sizeof(float))
        );
        tensor_protos[i * 4 + 3] = std::move(offset_initializer);
    }

    for (size_t i = 0; i < n_tensors; ++i) {
        const auto &tensor_name = model_weight.names[i];

        std::string base_dequant_name = tensor_name + "_base_dequant";

        auto *dequant_node = graph->add_node();
        dequant_node->set_op_type("DequantizeLinear");
        dequant_node->add_input(tensor_name + "_base");
        dequant_node->add_input(tensor_name + "_base_scale");
        dequant_node->add_input(tensor_name + "_base_zero_point");
        dequant_node->add_output(base_dequant_name);

        auto* base_offset_node = graph->add_node();
        base_offset_node->set_op_type("Add");
        base_offset_node->add_input(base_dequant_name);
        base_offset_node->add_input(tensor_name + "_base_offset");
        base_offset_node->add_output(tensor_name);
    }

    // Add all initializer to graph
    auto* initializer_list = graph->mutable_initializer();
    initializer_list->Reserve(static_cast<int>(tensor_protos.size()));

    for (const auto& initializer : tensor_protos) {
        if (initializer.name().empty()) {
            continue;
        }
        auto* init = initializer_list->Add();
        init->CopyFrom(initializer);
    }
    return model;
}


std::shared_ptr<Model> Compressor::reconstructModelUInt8WithDelta(
    const TensorPage &model_weight,
    const ModelStructure &model_structure
) const {
    // 1. Parse model structure
    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromArray(
        model_structure.serialized_model_structure.data(),
        static_cast<int>(model_structure.serialized_model_structure.size())
    )) {
        throw std::runtime_error("Compressor::reconstructModelUInt8WithDelta: Failed to parse model");
    }
    auto model = std::make_shared<Model>(onnx_model);

    auto *graph = model->model.mutable_graph();
    graph->clear_initializer();

    const size_t n_tensors = model_weight.names.size();
    std::vector<onnx::TensorProto> tensor_protos(n_tensors * 8);    // base, delta, 2x scale, 2x zero_point

    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < static_cast<int>(n_tensors); ++i) {
        const auto& tensor_name = model_weight.names[i];
        const auto &packet_ptr = model_weight.tensor_packets[i];

        if (auto int_packet = std::dynamic_pointer_cast<IntTensorPacket>(packet_ptr)) {
            // If the tensor is an integer type, we directly create a TensorF64 from IntTensorPacket.
            auto tensor = int_packet->toTensor64();
            onnx::TensorProto initializer;
            initializer.set_name(tensor_name);
            for (const auto &dim : tensor.getShape()) {
                initializer.add_dims(dim);
            }
            fillRawDataFromIntType(
                tensor.getTensor().data(),
                tensor.getDimension(),
                int_packet->getIntType(),
                initializer
            );
            tensor_protos[i * 8 + 0] = std::move(initializer);
            continue;
        }

        const auto &packet = *packet_ptr;
        const int64_t base_id = packet.getBaseTensorId();
        const int dimension = packet.getDimension();
        const int index_id = packet.getIndexId();

        auto index = index_cache_manager_->get(dimension, index_id);
        if (!index) {
            throw std::runtime_error("Cannot get index for dimension " + std::to_string(dimension));
        }

        auto base_packet = index->retrieveUINT8Quantized(base_id);
        index_cache_manager_->release(index, dimension, index_id, false);

        const auto& base_data = base_packet.data;
        auto base_scale = static_cast<float>(base_packet.scale);
        auto base_zero_point = static_cast<float>(base_packet.zero_point);
        // int8_t base_zero_point = saturateCastDoubleToInt8(base_packet.zero_point);
        // TensorType::VectorInt8 base_data_int8 = base_data.unaryExpr(SaturateU8ToI8());

        auto delta_packet_ptr = std::dynamic_pointer_cast<DeltaQuantPacket>(model_weight.tensor_packets[i]);
        if (!delta_packet_ptr) {
            throw std::runtime_error("Expected DeltaQuantPacket for tensor " + tensor_name);
        }
        auto delta_packet = delta_packet_ptr->toUINT8QuantizedTensorPacket();
        const auto& delta_data = delta_packet.data;
        auto delta_scale = static_cast<float>(delta_packet.scale);
        auto delta_zero_point = static_cast<float>(delta_packet.zero_point);
        int full_quantized_bit_width = delta_packet.full_quantized_bit_width;
        int discard_bits = full_quantized_bit_width - 8;
        if (discard_bits > 0) {
            float factor = std::pow(2.0f, static_cast<float>(discard_bits));
            delta_scale *= factor;
            delta_zero_point /= factor;
        }

        onnx::TensorProto base_initializer;
        base_initializer.set_name(tensor_name + "_base");
        base_initializer.set_data_type(onnx::TensorProto::UINT8);
        for (const auto& dim : packet.getShape()) {
            base_initializer.add_dims(dim);
        }
        base_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(base_data.data()), base_data.size())
        );
        tensor_protos[i * 8 + 0] = std::move(base_initializer);

        onnx::TensorProto delta_initializer;
        delta_initializer.set_name(tensor_name + "_delta");
        delta_initializer.set_data_type(onnx::TensorProto::UINT8);
        for (const auto& dim : packet.getShape()) {
            delta_initializer.add_dims(dim);
        }
        delta_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(delta_data.data()), delta_data.size())
        );
        tensor_protos[i * 8 + 1] = std::move(delta_initializer);

        onnx::TensorProto base_scale_initializer;
        base_scale_initializer.set_name(tensor_name + "_base_scale");
        base_scale_initializer.set_data_type(onnx::TensorProto::FLOAT);
        base_scale_initializer.add_dims(1);
        base_scale_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&base_scale), sizeof(float))
        );
        tensor_protos[i * 8 + 2] = std::move(base_scale_initializer);

        onnx::TensorProto base_zero_point_initializer;
        base_zero_point_initializer.set_name(tensor_name + "_base_zero_point");
        base_zero_point_initializer.set_data_type(onnx::TensorProto::UINT8);
        base_zero_point_initializer.add_dims(1);
        uint8_t zero_val = 0;
        base_zero_point_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&zero_val), sizeof(uint8_t))
        );
        tensor_protos[i * 8 + 3] = std::move(base_zero_point_initializer);

        onnx::TensorProto delta_scale_initializer;
        delta_scale_initializer.set_name(tensor_name + "_delta_scale");
        delta_scale_initializer.set_data_type(onnx::TensorProto::FLOAT);
        delta_scale_initializer.add_dims(1);
        delta_scale_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&delta_scale), sizeof(float))
        );
        tensor_protos[i * 8 + 4] = std::move(delta_scale_initializer);

        onnx::TensorProto delta_zero_point_initializer;
        delta_zero_point_initializer.set_name(tensor_name + "_delta_zero_point");
        delta_zero_point_initializer.set_data_type(onnx::TensorProto::UINT8);
        delta_zero_point_initializer.add_dims(1);
        uint8_t zero_val2 = 0;
        delta_zero_point_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&zero_val2), sizeof(uint8_t))
        );
        tensor_protos[i * 8 + 5] = std::move(delta_zero_point_initializer);

        // OFFSET
        float base_offset = - base_zero_point * base_scale;
        onnx::TensorProto base_offset_initializer;
        base_offset_initializer.set_name(tensor_name + "_base_offset");
        base_offset_initializer.set_data_type(onnx::TensorProto::FLOAT);
        base_offset_initializer.add_dims(1);
        base_offset_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&base_offset), sizeof(float))
        );
        tensor_protos[i * 8 + 6] = std::move(base_offset_initializer);

        float delta_offset = - delta_zero_point * delta_scale;
        onnx::TensorProto delta_offset_initializer;
        delta_offset_initializer.set_name(tensor_name + "_delta_offset");
        delta_offset_initializer.set_data_type(onnx::TensorProto::FLOAT);
        delta_offset_initializer.add_dims(1);
        delta_offset_initializer.set_raw_data(
            std::string(reinterpret_cast<const char*>(&delta_offset), sizeof(float))
        );
        tensor_protos[i * 8 + 7] = std::move(delta_offset_initializer);
    }

    for (size_t i = 0; i < n_tensors; ++i) {
        const auto& tensor_name = model_weight.names[i];

        std::string base_dequant_name = tensor_name + "_base_dequant";
        std::string delta_dequant_name = tensor_name + "_delta_dequant";

        auto* dequant_base_node = graph->add_node();
        dequant_base_node->set_op_type("DequantizeLinear");
        dequant_base_node->add_input(tensor_name + "_base");
        dequant_base_node->add_input(tensor_name + "_base_scale");
        dequant_base_node->add_input(tensor_name + "_base_zero_point");
        dequant_base_node->add_output(base_dequant_name);

        auto* dequant_delta_node = graph->add_node();
        dequant_delta_node->set_op_type("DequantizeLinear");
        dequant_delta_node->add_input(tensor_name + "_delta");
        dequant_delta_node->add_input(tensor_name + "_delta_scale");
        dequant_delta_node->add_input(tensor_name + "_delta_zero_point");
        dequant_delta_node->add_output(delta_dequant_name);

        // add offset: base
        auto* base_offset_node = graph->add_node();
        base_offset_node->set_op_type("Add");
        base_offset_node->add_input(base_dequant_name);
        base_offset_node->add_input(tensor_name + "_base_offset");
        base_offset_node->add_output(tensor_name + "_base_dequant_offset");

        // add offset: delta
        auto* delta_offset_node = graph->add_node();
        delta_offset_node->set_op_type("Add");
        delta_offset_node->add_input(delta_dequant_name);
        delta_offset_node->add_input(tensor_name + "_delta_offset");
        delta_offset_node->add_output(tensor_name + "_delta_dequant_offset");

        auto* add_node = graph->add_node();
        add_node->set_op_type("Add");
        add_node->add_input(tensor_name + "_base_dequant_offset");
        add_node->add_input(tensor_name + "_delta_dequant_offset");
        add_node->add_output(tensor_name);
    }

    // Add all initializer to graph
    auto* initializer_list = graph->mutable_initializer();
    initializer_list->Reserve(static_cast<int>(tensor_protos.size()));

    for (const auto& initializer : tensor_protos) {
        if (initializer.name().empty()) {
            continue;
        }
        auto* init = initializer_list->Add();
        init->CopyFrom(initializer);
    }
    return model;
}

/************************************ C ************************************/
// void cp_compress(
//     CompressorC *compressor,
//     ModelC *model,
//     ModelStructureC *model_structure_out,
//     TensorPageC *tensor_page_out
// ) {
//     auto [model_structure, tensor_page] = reinterpret_cast<Compressor *>(compressor)->compress(
//         *reinterpret_cast<Model *>(model)
//     );
//     *model_structure_out = model_structure;
//     *tensor_page_out = tensor_page;
// }
//
// void cp_decompress(
//     CompressorC *compressor,
//     ModelStructureC *model_structure,
//     TensorPageC *tensor_page,
//     ModelC *model_out
// ) {
//     auto model = reinterpret_cast<Compressor *>(compressor)->decompress(
//         *reinterpret_cast<ModelStructure *>(model_structure),
//         *reinterpret_cast<TensorPage *>(tensor_page)
//     );
//     *model_out = model;
// }
