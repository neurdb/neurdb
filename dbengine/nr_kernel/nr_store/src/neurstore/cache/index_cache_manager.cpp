#include "neurstore/cache/index_cache_manager.h"

#include <filesystem>

#include "neurstore/compress/index/router/single_index_router.h"


/**
 * Create a key for the index cache.
 * @param dimension
 * @param index_id
 */
static std::string makeKey(const int dimension, const int index_id) {
    return std::to_string(dimension) + "_" + std::to_string(index_id);
}

IndexCacheManager::IndexCacheManager(std::string store_path, int64_t max_bytes)
    : store_path_(std::move(store_path)), max_bytes_(max_bytes) {
    if (max_bytes_ <= 0) {
        throw std::invalid_argument("IndexCacheManager::IndexCacheManager: max_bytes must be positive");
    }
    if (!std::filesystem::exists(store_path_)) {
        std::filesystem::create_directories(store_path_);
    }
    if (!std::filesystem::exists(store_path_ + "/router")) {
        std::filesystem::create_directories(store_path_ + "/router");
    }
    loadRouters();
}

std::shared_ptr<VectorUInt8Index> IndexCacheManager::get(
    int dimension,
    int index_id
) {
    std::string key = makeKey(dimension, index_id);
    std::lock_guard lock(index_cache_mutex_);

    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        // Cache hit
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        use_count_map_[key]++;
        return it->second->second;
    }
    // cache miss
    std::shared_ptr<VectorUInt8Index> index;
    std::string index_path = store_path_ + "/" + key + ".index";

    if (std::filesystem::exists(index_path)) {
        std::ifstream file(index_path, std::ios::binary);
        if (file) index = VectorUInt8Index::deserialize(file);
    }
    if (!index) {
        int init_capacity = getInitialIndexCapacity(dimension, INDEX_RESIZE_TIMES);
        index = std::make_shared<VectorUInt8Index>(
            std::make_shared<QuantizedHNSWIndexHNSWLIB>(dimension, 0, init_capacity),
            32,
            dimension
        );
    }

    cache_list_.emplace_front(key, index);
    cache_map_[key] = cache_list_.begin();
    use_count_map_[key] = 1;

    int64_t space_bytes = index->space();
    size_map_[key] = space_bytes;
    used_bytes_ += space_bytes;
    updated_map_[key] = false;

    evictCacheIfNeeded();
    return index;
}

void IndexCacheManager::release(
    std::shared_ptr<VectorUInt8Index> index_ptr,
    int dimension,
    int index_id,
    bool is_updated
) {
    std::string key = makeKey(dimension, index_id);
    std::lock_guard lock(index_cache_mutex_);

    if (--use_count_map_[key] < 0) {
        use_count_map_[key] = 0;
    }
    if (is_updated) {
        int64_t new_space = index_ptr->space();
        used_bytes_ += (new_space - size_map_[key]);
        size_map_[key] = new_space;
        updated_map_[key] = true;
        evictCacheIfNeeded();
    }
    index_ptr.reset();
}

void IndexCacheManager::evictCacheIfNeeded() {
    while (used_bytes_ > max_bytes_) {
        for (auto it = cache_list_.rbegin(); it != cache_list_.rend(); ++it) {
            const std::string& evict_key = it->first;
            if (use_count_map_[evict_key] == 0) {
                if (updated_map_[evict_key]) {
                    saveIndex(evict_key, it->second);
                }
                updated_map_.erase(evict_key);
                used_bytes_ -= size_map_[evict_key];
                cache_map_.erase(evict_key);
                use_count_map_.erase(evict_key);
                size_map_.erase(evict_key);
                cache_list_.erase(std::next(it).base());
                break;
            }
        }
    }
}

void IndexCacheManager::markDirty(const std::string &key) {
    std::lock_guard lock(index_cache_mutex_);
    updated_map_[key] = true;
}

void IndexCacheManager::saveIndex(const std::string& key, const std::shared_ptr<VectorUInt8Index>& index) const {
    std::string index_path = store_path_ + "/" + key + ".index";
    std::ofstream file(index_path, std::ios::binary);
    if (!file) throw std::runtime_error("IndexCacheManager::saveIndex: Failed to open file " + index_path);
    index->serializeToStream(file);
}

void IndexCacheManager::saveIndexes(
    const std::list<std::pair<std::string, std::shared_ptr<VectorUInt8Index> > > &indexes
) const {
    for (const auto &[key, index]: indexes) {
        saveIndex(key, index);
    }
}

std::shared_ptr<VectorFloat64Router> IndexCacheManager::getRouter(int dimension) {
    std::lock_guard lock(router_mutex_);
    auto router = router_map_.find(dimension);
    if (router != router_map_.end()) {
        return router->second;
    }

    // no router, try load from disk
    std::string router_file = store_path_ + "/router/" + std::to_string(dimension) + ".router";
    if (std::filesystem::exists(router_file)) {
        std::ifstream in(router_file, std::ios::binary | std::ios::ate);
        if (!in) {
            throw std::runtime_error("IndexCacheManager::getRouter: Failed to open router file " + router_file);
        }

        std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("IndexCacheManager::getRouter: Failed to read router file " + router_file);
        }

        auto new_router = SingleIndex64Router::deserialize(buffer.data(), buffer.size());
        router_map_[dimension] = new_router;
        return new_router;
    }
    // no router found, create a new one
    auto new_router = std::make_shared<SingleIndex64Router>(
        dimension,
        getInitialIndexCapacity(dimension, INDEX_RESIZE_TIMES) << INDEX_RESIZE_TIMES
    );
    router_map_[dimension] = new_router;
    return new_router;
}

void IndexCacheManager::loadRouters() {
    std::lock_guard lock(router_mutex_);
    const std::string router_path = store_path_ + "/router";
    if (!std::filesystem::exists(router_path)) {
        std::filesystem::create_directories(router_path);
    }
    for (const auto &entry: std::filesystem::directory_iterator(router_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".router") {
            // load router into memory
            std::string dimension_str = entry.path().stem().string();
            int dimension = std::stoi(dimension_str);

            std::ifstream in(entry.path(), std::ios::binary | std::ios::ate);
            if (!in) {
                throw std::runtime_error(
                    "IndexCacheManager::loadRouters: Failed to open router file " + entry.path().string()
                );
            }

            std::streamsize size = in.tellg();
            in.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(size);
            if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
                throw std::runtime_error(
                    "IndexCacheManager::loadRouters: Failed to read router file " + entry.path().string()
                );
            }
            auto router = SingleIndex64Router::deserialize(buffer.data(), buffer.size());
            router_map_[dimension] = router;
        }
    }
}

void IndexCacheManager::saveRouters() const {
    const std::string router_path = store_path_ + "/router";
    if (!std::filesystem::exists(router_path)) {
        std::filesystem::create_directories(router_path);
    }

    for (const auto& [dimension, router] : router_map_) {
        const std::string router_file = router_path + "/" + std::to_string(dimension) + ".router";

        std::ostringstream oss(std::ios::binary);
        router->serialize(oss);
        const std::string& serialized_data = oss.str();

        std::ofstream out(router_file, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("IndexCacheManager::saveRouters: Failed to open file " + router_file);
        }
        out.write(serialized_data.data(), static_cast<std::streamsize>(serialized_data.size()));
        if (!out) {
            throw std::runtime_error("IndexCacheManager::saveRouters: Failed to write to file " + router_file);
        }
    }
}

void IndexCacheManager::clearCache() {
    saveRouters();
    std::lock_guard lock(index_cache_mutex_);
    for (auto it = cache_list_.begin(); it != cache_list_.end(); ++it) {
        const std::string& key = it->first;
        if (updated_map_[key]) {
            saveIndex(key, it->second);
        }
    }
}

IndexCacheManagerC* icm_create(const char *store_path) {
    std::string path(store_path);
    return reinterpret_cast<IndexCacheManagerC *>(new IndexCacheManager(path));
}

void icm_destroy(IndexCacheManagerC *icm) {
    auto *manager = reinterpret_cast<IndexCacheManager *>(icm);
    manager->clearCache();
    delete manager;
}

void icm_clear_cache(IndexCacheManagerC *icm) {
    auto *manager = reinterpret_cast<IndexCacheManager *>(icm);
    manager->clearCache();
}
