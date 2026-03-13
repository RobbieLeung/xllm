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

#include "sample_slot.h"

#include <string_view>

namespace xllm {

std::vector<size_t> find_literal_match_offsets(const std::string& prompt,
                                               const std::string& literal) {
  std::vector<size_t> offsets;
  if (literal.empty()) {
    return offsets;
  }

  size_t search_from = 0;
  while (true) {
    const size_t match_offset = prompt.find(literal, search_from);
    if (match_offset == std::string::npos) {
      return offsets;
    }
    offsets.push_back(match_offset);
    search_from = match_offset + literal.size();
  }
}

bool build_sample_slots(const std::string& request_id,
                        const std::string& prompt,
                        const std::string& literal,
                        const Tokenizer& tokenizer,
                        std::vector<SampleSlot>* sample_slots) {
  if (sample_slots == nullptr) {
    return false;
  }

  sample_slots->clear();
  const auto match_offsets = find_literal_match_offsets(prompt, literal);
  sample_slots->reserve(match_offsets.size());

  std::vector<int32_t> literal_tokens;
  if (!literal.empty() &&
      !tokenizer.encode(std::string_view(literal), &literal_tokens, false)) {
    return false;
  }

  for (size_t sample_id = 0; sample_id < match_offsets.size(); ++sample_id) {
    const size_t match_offset = match_offsets[sample_id];
    std::vector<int32_t> prefix_tokens;
    if (!tokenizer.encode(std::string_view(prompt.data(), match_offset),
                          &prefix_tokens,
                          false)) {
      return false;
    }

    SampleSlot slot;
    slot.request_id = request_id;
    slot.sequence_index = 0;
    slot.sample_id = sample_id;
    slot.token_position = prefix_tokens.size();
    slot.selector_match_offset = match_offset;
    slot.selector_token_count = literal_tokens.size();
    sample_slots->push_back(std::move(slot));
  }

  return true;
}

}  // namespace xllm
