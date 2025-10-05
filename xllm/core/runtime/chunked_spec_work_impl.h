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

#include "speculative_worker_impl.h"

namespace xllm {

class ChunkedSpecWorkerImpl : public SpeculativeWorkerImpl {
 public:
  ChunkedSpecWorkerImpl() = default;
  ~ChunkedSpecWorkerImpl() = default;

  std::optional<ForwardOutput> step(const ForwardInput& inputs) override;

 protected:
  std::optional<ForwardOutput> step_prefill(
      const ForwardInput& inputs) override;

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const ForwardInput& inputs,
                              ForwardInput& prefill_inputs) override;
};

}  // namespace xllm
