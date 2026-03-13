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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "request.h"
#include "request_state.h"

namespace xllm {
namespace {

class CharTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    if (ids == nullptr) {
      return false;
    }
    ids->clear();
    if (add_special_tokens) {
      ids->push_back(kBosTokenId);
    }
    for (char ch : text) {
      ids->push_back(static_cast<unsigned char>(ch));
    }
    return true;
  }

 private:
  static constexpr int32_t kBosTokenId = 1;
};

TEST(SampleSlotTest, BuildSampleSlotsKeepsMatchOrderAndSampleIds) {
  CharTokenizer tokenizer;
  std::vector<SampleSlot> sample_slots;

  ASSERT_TRUE(build_sample_slots("sample-req",
                                 "A<emb_0>B<emb_0>C",
                                 "<emb_0>",
                                 tokenizer,
                                 &sample_slots));

  ASSERT_EQ(sample_slots.size(), 2);

  EXPECT_EQ(sample_slots[0].request_id, "sample-req");
  EXPECT_EQ(sample_slots[0].sequence_index, 0);
  EXPECT_EQ(sample_slots[0].sample_id, 0);
  EXPECT_EQ(sample_slots[0].token_position, 1);
  EXPECT_EQ(sample_slots[0].selector_match_offset, 1);
  EXPECT_EQ(sample_slots[0].selector_token_count, 7);

  EXPECT_EQ(sample_slots[1].sample_id, 1);
  EXPECT_EQ(sample_slots[1].token_position, 9);
  EXPECT_EQ(sample_slots[1].selector_match_offset, 9);
  EXPECT_EQ(sample_slots[1].selector_token_count, 7);
}

TEST(SampleSlotTest, RequestPropagatesSampleSlotsToSequenceRuntime) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  RequestState request_state("abc",
                             std::vector<int32_t>{10, 11, 12},
                             sampling_param,
                             SchedulerParam{},
                             stopping_checker,
                             /*seq_capacity=*/8,
                             /*n=*/1,
                             /*best_of=*/1,
                             /*logprobs=*/false,
                             /*stream=*/false,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true,
                             /*enable_schedule_overlap=*/false,
                             [](const RequestOutput&) { return true; },
                             OutputsFunc{});

  SampleSlot first_slot;
  first_slot.request_id = "sample-req";
  first_slot.sequence_index = 0;
  first_slot.sample_id = 0;
  first_slot.token_position = 2;
  first_slot.selector_match_offset = 2;
  first_slot.selector_token_count = 1;

  SampleSlot second_slot;
  second_slot.request_id = "sample-req";
  second_slot.sequence_index = 0;
  second_slot.sample_id = 1;
  second_slot.token_position = 4;
  second_slot.selector_match_offset = 4;
  second_slot.selector_token_count = 1;

  request_state.sample_slots = {first_slot, second_slot};

  Request request("sample-req", "", "", request_state);

  ASSERT_EQ(request.sequences().size(), 1);
  const auto& runtime_sample_slots = request.sequences()[0]->sample_slots();
  ASSERT_EQ(runtime_sample_slots.size(), 2);
  EXPECT_EQ(runtime_sample_slots[0].sample_id, 0);
  EXPECT_EQ(runtime_sample_slots[0].token_position, 2);
  EXPECT_EQ(runtime_sample_slots[1].sample_id, 1);
  EXPECT_EQ(runtime_sample_slots[1].selector_match_offset, 4);
}

}  // namespace
}  // namespace xllm
