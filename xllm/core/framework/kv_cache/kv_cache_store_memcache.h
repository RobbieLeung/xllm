#pragma once
#include <memcache/cpp/mmcache.h>

#include "kv_cache_store.h"

namespace xllm {

class MemcacheStore final : public KVCacheStore {
 public:
  MemcacheStore(const StoreConfig& config,
                std::vector<xllm::KVCache>* host_kv_caches);
  ~MemcacheStore() = default;
  MemcacheStore(const MemcacheStore&) = delete;
  MemcacheStore& operator=(const MemcacheStore&) = delete;

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_remove(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_exist(std::vector<std::string>&& keys) override;

 private:
  std::shared_ptr<ock::mmc::ObjectStore> obj_store_;
};

}  // namespace xllm
