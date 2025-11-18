#pragma once

#include <Mooncake/mooncake-store/include/client.h>

#include "kv_cache_store.h"

namespace xllm {

class MooncakeStore final : public KVCacheStore {
 public:
  MooncakeStore(const StoreConfig& config,
                std::vector<xllm::KVCache>* host_kv_caches);
  ~MooncakeStore();
  MooncakeStore(const MooncakeStore&) = delete;
  MooncakeStore& operator=(const MooncakeStore&) = delete;

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_remove(Slice<BlockTransferInfo>& block_transfer_info) override;

  uint32_t batch_exist(std::vector<std::string>&& keys) override;

 private:
  mooncake::ReplicateConfig rep_config_;
  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
