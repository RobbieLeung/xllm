
#include "kv_cache_store_memcache.h"

#include "util/hash_util.h"

namespace xllm {
typedef enum {
  SMEMB_COPY_L2G = 0, /* copy data from local space to global space */
  SMEMB_COPY_G2L = 1, /* copy data from global space to local space */
  SMEMB_COPY_G2H = 2, /* copy data from global space to host memory */
  SMEMB_COPY_H2G = 3, /* copy data from host memory to global space */
  SMEMB_COPY_G2G = 4, /* copy data from global space to global space */
  /* add here */
  SMEMB_COPY_BUTT
} smem_bm_copy_type;

MemcacheStore::MemcacheStore(const StoreConfig& config,
                             std::vector<xllm::KVCache>* host_kv_caches)
    : KVCacheStore(config, host_kv_caches) {
  obj_store_ = ock::mmc::ObjectStore::CreateObjectStore();
  obj_store_->Init(config_.device_idx);

  if (config_.total_size > 0 && config_.tensor_data != nullptr) {
    auto register_result =
        obj_store_->RegisterBuffer(config_.tensor_data, config_.total_size);

    if (register_result != 0) {
      LOG(ERROR) << "Failed to register host memory, error code: "
                 << register_result;
      return;
    }
  } else {
    LOG(FATAL) << "MemcacheStore must RegisterBuffer, but got register size: "
               << config_.total_size
               << ", and data ptr: " << uint64_t(config_.tensor_data);
  }
  LOG(INFO) << "Success create MemcacheStore!";
}

uint32_t MemcacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<void*>> buffers;
  std::vector<std::vector<size_t>> sizes;

  str_keys.reserve(block_transfer_info.size());
  buffers.reserve(block_transfer_info.size());
  sizes.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    if (obj_store_->IsExist(str_key) == 0) {
      continue;
    }
    str_keys.emplace_back(std::move(str_key));

    void* k_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    void* v_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    buffers.emplace_back(std::vector<void*>{k_cache, v_cache});
    sizes.emplace_back(
        std::vector<size_t>{k_cache_size_per_block_, v_cache_size_per_block_});
  }

  if (str_keys.size() == 0) {
    return block_transfer_info.size();
  }

  uint64_t success_cnt = str_keys.size();
  auto results =
      obj_store_->BatchPutFromLayers(str_keys, buffers, sizes, SMEMB_COPY_L2G);

  for (int i = 0; i < str_keys.size(); i++) {
    if (results[i] != 0) {
      success_cnt = i;
      DLOG(ERROR) << "success_cnt: " << success_cnt
                  << ", failed to BatchPutFromLayers, error code: "
                  << results[i];
      break;
    }
  }
  return success_cnt;
}

uint32_t MemcacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<void*>> buffers;
  std::vector<std::vector<size_t>> sizes;

  str_keys.reserve(block_transfer_info.size());
  buffers.reserve(block_transfer_info.size());
  sizes.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    if (obj_store_->IsExist(str_key) == 0) {
      break;
    }
    str_keys.emplace_back(std::move(str_key));

    void* k_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    void* v_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    buffers.emplace_back(std::vector<void*>{k_cache, v_cache});
    sizes.emplace_back(
        std::vector<size_t>{k_cache_size_per_block_, v_cache_size_per_block_});
  }

  if (str_keys.size() == 0) {
    return 0;
  }

  uint64_t success_cnt = str_keys.size();

  auto results =
      obj_store_->BatchGetIntoLayers(str_keys, buffers, sizes, SMEMB_COPY_G2L);
  for (int i = 0; i < str_keys.size(); i++) {
    if (results[i] != 0) {
      success_cnt = i;
      DLOG(ERROR) << "success_cnt: " << success_cnt
                  << ", failed to BatchGetIntoLayers, error code: "
                  << results[i];
      break;
    }
  }
  return success_cnt;
}

uint32_t MemcacheStore::batch_remove(
    Slice<BlockTransferInfo>& block_transfer_info) {
  uint64_t success_cnt = 0;
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    auto result = obj_store_->Remove(str_key);

    if (result == 0) {
      success_cnt++;
    }
  }
  return success_cnt;
}

uint32_t MemcacheStore::batch_exist(std::vector<std::string>&& keys) {
  auto exist_vec = obj_store_->BatchIsExist(std::move(keys));
  uint32_t ret = 0;
  for (auto exist : exist_vec) {
    if (exist == 0) {
      break;
    }
    ret++;
  }
  return ret;
}

}  // namespace xllm
