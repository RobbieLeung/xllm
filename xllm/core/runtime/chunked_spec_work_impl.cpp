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

#pragma once

#include "chunked_spec_work_impl.h"

namespace xllm {
void ChunkedSpecWorkerImpl::prepare_prefill_inputs(
    const ForwardInput& inputs,
    ForwardInput& prefill_inputs) {
  prefill_inputs = inputs.to(device_, dtype_);
  auto& input_params = prefill_inputs.input_params;

  torch::Tensor token_ids = safe_to(inputs.token_ids, torch::kCPU);
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     inputs.token_ids.numel()};

  auto& extra_token_ids = input_params.extra_token_ids;
  //   auto& kv_seq_lens_vec = input_params.kv_seq_lens_vec;
  //   auto& q_seq_lens_vec = input_params.q_seq_lens_vec;

  //   CHECK_EQ(kv_seq_lens_vec.size(), input_params.num_sequences);
  //   CHECK_EQ(q_seq_lens_vec.size(), input_params.num_sequences);

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(inputs.token_ids.numel());
  for (size_t i = 0; i < input_params.num_sequences; ++i) {
    int32_t q_len = 0;
    q_len = input_params.q_seq_lens_vec[i];
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_inputs.token_ids =
      torch::tensor(new_token_ids, prefill_inputs.positions.options());
}
}  // namespace xllm
