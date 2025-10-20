#include "neurstore/utils/model.h"

#include <fstream>

#include "neurstore/compress/method/delta_quant_compress.h"
#include "neurstore/utils/tensor.h"


/*********************************** C++ ***********************************/
/* Model */
Model::Model() {
    this->model = onnx::ModelProto();
}

Model::Model(const onnx::ModelProto &model) {
    this->model = model;
}

Model::Model(onnx::ModelProto&& model) {
    this->model = std::move(model);
}

Model::Model(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Model::Model: failed to open file " + std::string(path));
    }
    if (!this->model.ParseFromIstream(&file)) {
        // ONNX
        throw std::runtime_error("Model::Model: failed to parse model");
    }
}

Model::Model(const std::vector<uint8_t> &serialized_data) {
    std::string model_str(serialized_data.begin(), serialized_data.end());
    if (!this->model.ParseFromString(model_str)) {
        throw std::runtime_error("Model::Model: failed to parse model");
    }
}

std::vector<uint8_t> Model::serialize() const {
    std::string serialized_model;
    if (!model.SerializeToString(&serialized_model)) {
        throw std::runtime_error("Model::serialize: failed to serialize model");
    }
    return std::vector<uint8_t>(serialized_model.begin(), serialized_model.end());
}

/* ModelWeight */
ModelWeight::ModelWeight(
    const std::vector<std::shared_ptr<TensorF64> > &tensors,
    const std::vector<std::string> &names,
    const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
) {
    if (tensors.size() != names.size()) {
        throw std::invalid_argument("ModuleWeight::ModuleWeight: tensors and names must have the same size");
    }
    if (int_types.size() != tensors.size()) {
        throw std::invalid_argument("ModuleWeight::ModuleWeight: int_types and tensors must have the same size");
    }
    this->tensors = tensors;
    this->names = names;
    this->int_types = int_types;
}

std::shared_ptr<TensorF64> ModelWeight::getTensorF64(uint index) const {
    if (index >= tensors.size()) {
        throw std::out_of_range("ModelWeight::getTensor: index out of range");
    }
    return tensors[index];
}

std::string ModelWeight::getName(uint index) const {
    if (index >= names.size()) {
        throw std::out_of_range("ModelWeight::getName: index out of range");
    }
    return names[index];
}

void ModelWeight::set(
    const std::vector<std::string> &names,
    const std::vector<std::shared_ptr<TensorF64> > &tensors,
    const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
) {
    if (names.size() != tensors.size()) {
        throw std::invalid_argument("ModelWeight::set: names and tensors must have the same size");
    }
    if (int_types.size() != tensors.size()) {
        throw std::invalid_argument("ModelWeight::set: int_types and tensors must have the same size");
    }
    this->names = names;
    this->tensors = tensors;
    this->int_types = int_types;
}

uint ModelWeight::nTensors() const {
    return tensors.size();
}

bool ModelWeight::isIntType(uint index) const {
    if (index >= int_types.size()) {
        throw std::out_of_range("ModelWeight::isIntType: index out of range");
    }
    return int_types[index].has_value();
}

IntTensorPacket::IntType ModelWeight::getIntType(uint index) const {
    if (index >= int_types.size()) {
        throw std::out_of_range("ModelWeight::getIntType: index out of range");
    }
    if (!int_types[index].has_value()) {
        throw std::invalid_argument("ModelWeight::getIntType: index is not an int type");
    }
    return int_types[index].value();
}

/* ModelWeightF16 */
ModelWeightF16::ModelWeightF16(
    const std::vector<std::shared_ptr<TensorF16> > &tensor_f16,
    const std::vector<std::string> &names,
    const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
) {
    if (tensor_f16.size() != names.size()) {
        throw std::invalid_argument("ModelWeightF16::ModelWeightF16: tensors and names must have the same size");
    }
    this->tensor_f16 = tensor_f16;
    this->names = names;
    this->int_types = int_types;
}

std::shared_ptr<TensorF16> ModelWeightF16::getTensorF16(uint index) const {
    if (index >= tensor_f16.size()) {
        throw std::out_of_range("ModelWeightF16::getTensor: index out of range");
    }
    return tensor_f16[index];
}

std::string ModelWeightF16::getName(uint index) const {
    if (index >= names.size()) {
        throw std::out_of_range("ModelWeightF16::getName: index out of range");
    }
    return names[index];
}

void ModelWeightF16::set(
    const std::vector<std::string> &names,
    const std::vector<std::shared_ptr<TensorF16> > &tensor_f16,
    const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
) {
    if (names.size() != tensor_f16.size()) {
        throw std::invalid_argument("ModelWeightF16::set: names and tensors must have the same size");
    }
    if (int_types.size() != tensor_f16.size()) {
        throw std::invalid_argument("ModelWeightF16::set: int_types and tensors must have the same size");
    }
    this->names = names;
    this->tensor_f16 = tensor_f16;
    this->int_types = int_types;
}

uint ModelWeightF16::nTensors() const {
    return tensor_f16.size();
}

bool ModelWeightF16::isIntType(uint index) const {
    if (index >= int_types.size()) {
        throw std::out_of_range("ModelWeightF16::isIntType: index out of range");
    }
    return int_types[index].has_value();
}

IntTensorPacket::IntType ModelWeightF16::getIntType(uint index) const {
    if (index >= int_types.size()) {
        throw std::out_of_range("ModelWeightF16::getIntType: index out of range");
    }
    if (!int_types[index].has_value()) {
        throw std::invalid_argument("ModelWeightF16::getIntType: index is not an int type");
    }
    return int_types[index].value();
}

/* TensorPage */
TensorPage::TensorPage(
    const std::vector<std::shared_ptr<TensorPacket> > &tensor_packets,
    const std::vector<std::string> &names
) {
    if (tensor_packets.size() != names.size()) {
        throw std::invalid_argument(
            "TensorPage::TensorPage: tensors and ids must have the same size");
    }
    this->tensor_packets = tensor_packets;
    this->names = names;
}

TensorPage::TensorPage() {
    this->tensor_packets = {};
    this->names = {};
}

std::shared_ptr<TensorPacket> TensorPage::getTensorPacket(uint index) const {
    if (index >= tensor_packets.size()) {
        throw std::out_of_range("TensorPage::getTensorPacket: index out of range");
    }
    return tensor_packets[index];
}

std::string TensorPage::getName(uint index) const {
    if (index >= names.size()) {
        throw std::out_of_range("TensorPage::getName: index out of range");
    }
    return names[index];
}

void TensorPage::set(
    const std::vector<std::string> &names,
    const std::vector<std::shared_ptr<TensorPacket> > &tensor_packets
) {
    if (names.size() != tensor_packets.size()) {
        throw std::invalid_argument("TensorPage::set: ids and tensors must have the same size");
    }
    this->names = names;
    this->tensor_packets = tensor_packets;
}

uint TensorPage::nTensors() const {
    return tensor_packets.size();
}

std::vector<uint8_t> TensorPage::serialize() const {
    uint32_t packet_count = tensor_packets.size();
    size_t total_size = 0;
    total_size += sizeof(packet_count);

    std::vector<std::vector<uint8_t>> serialized_packets;
    serialized_packets.reserve(packet_count);
    for (const auto &tensor_packet: tensor_packets) {
        auto serialized_packet = tensor_packet->serialize();
        total_size += sizeof(uint32_t) + serialized_packet.size();
        serialized_packets.push_back(std::move(serialized_packet));
    }

    for (const auto &name : names) {
        total_size += sizeof(uint32_t);
        total_size += name.size();
    }

    std::vector<uint8_t> buffer(total_size);
    uint8_t *ptr = buffer.data();

    std::memcpy(ptr, &packet_count, sizeof(packet_count));
    ptr += sizeof(packet_count);

    for (size_t i = 0; i < tensor_packets.size(); i++) {
        const auto &sp = serialized_packets[i];
        uint32_t packet_size = sp.size();
        std::memcpy(ptr, &packet_size, sizeof(packet_size));
        ptr += sizeof(packet_size);
        std::memcpy(ptr, sp.data(), sp.size());
        ptr += sp.size();
    }

    for (const auto &name : names) {
        uint32_t name_size = name.size();
        std::memcpy(ptr, &name_size, sizeof(name_size));
        ptr += sizeof(name_size);

        std::memcpy(ptr, name.data(), name_size);
        ptr += name_size;
    }
    return buffer;
}

TensorPage TensorPage::deserialize(const std::vector<uint8_t> &buffer) {
    const uint8_t *ptr = buffer.data();
    size_t size = buffer.size();
    return deserialize(ptr, size);
}

TensorPage TensorPage::deserialize(const uint8_t* data, size_t size) {
    const uint8_t* ptr = data;

    uint32_t packet_count;
    std::memcpy(&packet_count, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    std::vector<std::shared_ptr<TensorPacket>> tensor_packets;
    tensor_packets.reserve(packet_count);
    for (uint32_t i = 0; i < packet_count; ++i) {
        uint32_t packet_size;
        std::memcpy(&packet_size, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        std::vector<uint8_t> packet_data(ptr, ptr + packet_size);
        ptr += packet_size;
        tensor_packets.push_back(DeltaQuantPacket::deserialize(packet_data));
    }

    std::vector<std::string> names;
    names.reserve(packet_count);
    for (uint32_t i = 0; i < packet_count; ++i) {
        uint32_t name_size;
        std::memcpy(&name_size, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        names.emplace_back(reinterpret_cast<const char*>(ptr), name_size);
        ptr += name_size;
    }
    return TensorPage(tensor_packets, names);
}

/* ModelStructure */
ModelStructure::ModelStructure(
    const std::vector<uint8_t> &serialized_data,
    const std::vector<std::string> &names
) {
    this->serialized_model_structure = serialized_data;
    this->names = names;
}

ModelStructure::ModelStructure() {
    this->serialized_model_structure = {};
    this->names = {};
}

void ModelStructure::setTensorNames(const std::vector<std::string> &names) {
    this->names = names;
}

void ModelStructure::setSerializedModelStructure(const std::vector<uint8_t> &serialized_data) {
    this->serialized_model_structure = serialized_data;
}

void ModelStructure::addTensorNames(const std::vector<std::string> &names) {
    this->names.insert(this->names.end(), names.begin(), names.end());
}

std::vector<uint8_t> ModelStructure::serialize() const {
    uint32_t structure_size = serialized_model_structure.size();
    uint32_t names_count = names.size();
    size_t total_size = 0;
    // structure_size (4B) + actual structure bytes
    total_size += sizeof(uint32_t) + structure_size;
    // names_count (4B)
    total_size += sizeof(uint32_t);
    // each name: (4B for name_size + actual chars)
    for (const auto &name : names) {
        total_size += sizeof(uint32_t);
        total_size += name.size();
    }
    std::vector<uint8_t> buffer(total_size);

    uint8_t *ptr = buffer.data();
    std::memcpy(ptr, &structure_size, sizeof(structure_size));
    ptr += sizeof(structure_size);

    std::memcpy(ptr, serialized_model_structure.data(), structure_size);
    ptr += structure_size;

    std::memcpy(ptr, &names_count, sizeof(names_count));
    ptr += sizeof(names_count);

    for (const auto &name : names) {
        uint32_t name_size = name.size();
        std::memcpy(ptr, &name_size, sizeof(name_size));
        ptr += sizeof(name_size);
        std::memcpy(ptr, name.data(), name_size);
        ptr += name_size;
    }
    return buffer;
}

ModelStructure ModelStructure::deserialize(const std::vector<uint8_t> &data) {
    const uint8_t *ptr = data.data();
    size_t size = data.size();
    return deserialize(ptr, size);
}

ModelStructure ModelStructure::deserialize(const uint8_t* data, size_t size) {
    const uint8_t* ptr = data;

    uint32_t structure_size;
    std::memcpy(&structure_size, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    std::vector<uint8_t> serialized_model_structure(ptr, ptr + structure_size);
    ptr += structure_size;

    uint32_t names_count;
    std::memcpy(&names_count, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    std::vector<std::string> names;
    names.reserve(names_count);
    for (uint32_t i = 0; i < names_count; i++) {
        uint32_t name_size;
        std::memcpy(&name_size, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        names.emplace_back(reinterpret_cast<const char*>(ptr), name_size);
        ptr += name_size;
    }

    return ModelStructure(serialized_model_structure, names);
}

/************************************ C ************************************/
/* Model */
ModelC *m_create_empty_model() {
    return reinterpret_cast<ModelC *>(new Model());
}

ModelC *m_create_model_from_path(const char *path) {
    return reinterpret_cast<ModelC *>(new Model(path));
}

ModelC *m_create_model_from_serialized(const uint8_t *data, size_t n) {
    std::vector<uint8_t> data_vec(data, data + n);
    return reinterpret_cast<ModelC *>(new Model(data_vec));
}

void m_destroy_model(ModelC *model) {
    delete reinterpret_cast<Model *>(model);
}

char *m_serialize(ModelC *model, size_t *out_size) {
    auto serialized = reinterpret_cast<Model *>(model)->serialize();
    char *buffer = new char[serialized.size()];
    std::memcpy(buffer, serialized.data(), serialized.size());
    *out_size = serialized.size();
    return buffer;
}

/* ModelWeight */
ModelWeightC *mw_create_model_weight() {
    return reinterpret_cast<ModelWeightC *>(new ModelWeight({}, {}, {}));
}

void mw_destroy_model_weight(ModelWeightC *model_weight) {
    delete reinterpret_cast<ModelWeight *>(model_weight);
}

uint mw_n_tensors(ModelWeightC *model_weight) {
    return reinterpret_cast<ModelWeight *>(model_weight)->nTensors();
}

const char *mw_get_name(ModelWeightC *model_weight, uint index) {
    return reinterpret_cast<ModelWeight *>(model_weight)->getName(index).c_str();
}

Tensor64C *mw_get_tensor(ModelWeightC *model_weight, uint index) {
    auto tensor = reinterpret_cast<ModelWeight *>(model_weight)->getTensorF64(index);
    return tensor ? new TensorF64(*tensor) : nullptr;
}

// void mw_set(ModelWeightC *model_weight, const char **names, Tensor64C **tensors, uint n) {
//     std::vector<std::string> names_vec;
//     std::vector<std::shared_ptr<Tensor64C> > tensors_vec;
//     for (uint i = 0; i < n; i++) {
//         names_vec.emplace_back(names[i]);
//         tensors_vec.push_back(std::shared_ptr<Tensor64C>(reinterpret_cast<Tensor64C *>(tensors[i])));
//     }
//     reinterpret_cast<ModelWeight *>(model_weight)->set(names_vec, tensors_vec);
// }

/* TensorPage */
TensorPageC *tp_create_tensor_page() {
    return reinterpret_cast<TensorPageC *>(new TensorPage({}, {}));
}

void tp_destroy_tensor_page(TensorPageC *tensor_page) {
    delete reinterpret_cast<TensorPage *>(tensor_page);
}

uint tp_n_tensors(TensorPageC *tensor_page) {
    return reinterpret_cast<TensorPage *>(tensor_page)->nTensors();
}

const char *tp_get_name(TensorPageC *tensor_page, uint index) {
    return reinterpret_cast<TensorPage *>(tensor_page)->getName(index).c_str();
}

TensorPacketC *tp_get_tensor_packet(TensorPageC *tensor_page, uint index) {
    return reinterpret_cast<TensorPacketC *>(
        reinterpret_cast<TensorPage *>(tensor_page)->getTensorPacket(index).get()
    );
}

void tp_set(
    TensorPageC *tensor_page,
    const char **names,
    TensorPacketC **tensor_packets,
    uint n
) {
    std::vector<std::string> names_vec;
    std::vector<std::shared_ptr<TensorPacketC> > tensor_packets_vec;
    for (uint i = 0; i < n; i++) {
        names_vec.emplace_back(names[i]);
        tensor_packets_vec.push_back(
            std::shared_ptr<TensorPacketC>(reinterpret_cast<TensorPacketC *>(tensor_packets[i]))
        );
    }
    reinterpret_cast<TensorPage *>(tensor_page)->set(names_vec, tensor_packets_vec);
}

char *tp_serialize(TensorPageC *tensor_page, size_t *out_size) {
    auto serialized = reinterpret_cast<TensorPage *>(tensor_page)->serialize();
    char *buffer = new char[serialized.size()];
    std::memcpy(buffer, serialized.data(), serialized.size());
    *out_size = serialized.size();
    return buffer;
}

TensorPageC *tp_deserialize(const uint8_t *data, size_t n) {
    return reinterpret_cast<TensorPageC *>(
        new TensorPage(TensorPage::deserialize(data, n))
    );
}

/* ModelStructure */
ModelStructureC *ms_create_model_structure() {
    return reinterpret_cast<ModelStructureC *>(new ModelStructure({}, {}));
}

void ms_destroy_model_structure(ModelStructureC *model_structure) {
    delete reinterpret_cast<ModelStructure *>(model_structure);
}

void ms_set_tensor_names(ModelStructureC *model_structure, const char **names, uint n) {
    std::vector<std::string> names_vec;
    for (uint i = 0; i < n; i++) {
        names_vec.emplace_back(names[i]);
    }
    reinterpret_cast<ModelStructure *>(model_structure)->setTensorNames(names_vec);
}

void ms_set_serialized_model_structure(ModelStructureC *model_structure, const uint8_t *data, uint n) {
    std::vector<uint8_t> data_vec;
    for (uint i = 0; i < n; i++) {
        data_vec.push_back(data[i]);
    }
    reinterpret_cast<ModelStructure *>(model_structure)->setSerializedModelStructure(data_vec);
}

void ms_add_tensor_names(ModelStructureC *model_structure, const char **names, uint n) {
    std::vector<std::string> names_vec;
    for (uint i = 0; i < n; i++) {
        names_vec.emplace_back(names[i]);
    }
    reinterpret_cast<ModelStructure *>(model_structure)->addTensorNames(names_vec);
}

char *ms_serialize(ModelStructureC *model_structure, size_t *out_size) {
    auto serialized = reinterpret_cast<ModelStructure *>(model_structure)->serialize();
    char *buffer = new char[serialized.size()];
    std::memcpy(buffer, serialized.data(), serialized.size());
    *out_size = serialized.size();
    return buffer;
}

ModelStructureC *ms_deserialize(const uint8_t *data, size_t n) {
    return reinterpret_cast<ModelStructureC *>(new ModelStructure(ModelStructure::deserialize(data, n)));
}
