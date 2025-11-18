
#include "kv_cache_store.h"

#include <glog/logging.h>

#include "kv_cache_store_memcache.h"
#include "kv_cache_store_mooncake.h"
namespace xllm {

KVCacheStore::KVCacheStore(const StoreConfig& config,
                           std::vector<xllm::KVCache>* host_kv_caches)
    : config_(config), host_kv_caches_(host_kv_caches) {
  auto k_tensor_one_block = host_kv_caches_->at(0).get_k_cache();
  auto v_tensor_one_block = host_kv_caches_->at(0).get_v_cache();

  k_cache_size_per_block_ =
      k_tensor_one_block.numel() * k_tensor_one_block.element_size();
  v_cache_size_per_block_ =
      v_tensor_one_block.numel() * v_tensor_one_block.element_size();

  LOG(INFO) << "k_cache_size_per_block: " << k_cache_size_per_block_;
  LOG(INFO) << "v_cache_size_per_block: " << v_cache_size_per_block_;
}

std::shared_ptr<KVCacheStore> create_kvcache_store(
    const StoreConfig& config,
    std::vector<xllm::KVCache>* host_kv_caches) {
  if (config.store_type == "mooncake") {
    return std::make_shared<MooncakeStore>(config, host_kv_caches);
  } else if (config.store_type == "memcache") {
    return std::make_shared<MemcacheStore>(config, host_kv_caches);
  } else {
    LOG(ERROR) << "Unrecgnized store type: " << config.store_type;
    return nullptr;
  }
}

}  // namespace xllm
