/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "embedding_allocator2.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace xllm {

EmbeddingAllocator2::EmbeddingAllocator2(int32_t total_embeddings,
                                         int32_t embedding_dim,
                                         torch::ScalarType dtype)
    : num_free_embeddings_(total_embeddings) {
  CHECK_GT(total_embeddings, 0) << "No embeddings to allocate";

  embeddings_cache_.resize(total_embeddings);
  free_embeddings_.reserve(total_embeddings);
  for (int32_t i = 0; i < total_embeddings; ++i) {
    free_embeddings_.push_back(total_embeddings - i - 1);
  }
}

EmbeddingAllocator2::~EmbeddingAllocator2() {
  CHECK(num_free_embeddings_ == free_embeddings_.size())
      << "Not all embeddings have been freed";
}

// allocate a embedding id
int32_t EmbeddingAllocator2::allocate() {
  CHECK(num_free_embeddings_ > 0) << "No more embeddings available";
  const int32_t embedding_id = free_embeddings_[--num_free_embeddings_];
  return embedding_id;
}

// caller should make sure the embedding_id is valid
void EmbeddingAllocator2::free(int32_t embedding_id) {
  CHECK(num_free_embeddings_ < free_embeddings_.size());
  free_embeddings_[num_free_embeddings_++] = embedding_id;
}

// write embeddings to cache
void EmbeddingAllocator2::write(int32_t embedding_id,
                                const std::vector<int32_t>& token_ids,
                                const torch::Tensor& embeddings) {
  embeddings_cache_[embedding_id].token_ids = std::move(token_ids);
  if (embeddings.dim() == 1) {
    embeddings_cache_[embedding_id].embeddings = embeddings.unsqueeze(0);
  } else {
    embeddings_cache_[embedding_id].embeddings = embeddings;
  }
}

void EmbeddingAllocator2::write(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::vector<int32_t>>& token_ids,
    const torch::Tensor& embeddings) {
  int32_t total_embeddings = embedding_ids.size();
  CHECK_EQ(total_embeddings, embeddings.size(0));
  for (int32_t i = 0; i < total_embeddings; ++i) {
    write(embedding_ids[i], token_ids[i], embeddings[i]);
  }
}

void EmbeddingAllocator2::write(const std::vector<int32_t>& embedding_ids,
                                const torch::Tensor& token_ids,
                                const torch::Tensor& embeddings) {
  int32_t num_sequences = embedding_ids.size();
  for (int32_t i = 0; i < num_sequences; ++i) {
    write(embedding_ids[i], {token_ids[i].item<int64_t>()}, embeddings[i]);
  }
}

void EmbeddingAllocator2::write_validate(
    const std::vector<int32_t>& embedding_ids,
    torch::Tensor&& next_tokens,
    const torch::Tensor& embeddings) {
  int32_t num_sequences = embedding_ids.size();
  for (int32_t i = 0; i < num_sequences; ++i) {
    torch::Tensor cur_tokens = next_tokens[i];
    int32_t token_idx = 0;
    std::vector<int32_t> valid_token_ids;
    for (; token_idx < cur_tokens.size(0); ++token_idx) {
      if (cur_tokens[token_idx].item<int32_t>() >= 0) {
        valid_token_ids.push_back(cur_tokens[token_idx].item<int32_t>());
      } else {
        break;
      }
    }
    write(embedding_ids[i],
          valid_token_ids,
          embeddings[i].narrow(0, 0, token_idx));
  }
}

// read embeddings from cache
torch::Tensor EmbeddingAllocator2::read(int32_t embedding_id) {
  return embeddings_cache_[embedding_id].embeddings;
}

torch::Tensor EmbeddingAllocator2::read(
    const std::vector<int32_t>& embedding_ids) {
  std::vector<torch::Tensor> embeddings;
  int32_t total_embeddings = embedding_ids.size();
  embeddings.reserve(total_embeddings);
  for (int32_t i = 0; i < total_embeddings; ++i) {
    embeddings.emplace_back(embeddings_cache_[embedding_ids[i]].embeddings);
  }
  return torch::cat(embeddings, 0);
}

std::vector<std::vector<int32_t>> EmbeddingAllocator2::read_token_ids(
    const std::vector<int32_t>& embedding_ids) {
  std::vector<std::vector<int32_t>> token_ids;
  int32_t total_embeddings = embedding_ids.size();
  token_ids.reserve(total_embeddings);
  for (int32_t i = 0; i < total_embeddings; ++i) {
    token_ids.emplace_back(embeddings_cache_[embedding_ids[i]].token_ids);
  }
  return token_ids;
}

}  // namespace xllm
