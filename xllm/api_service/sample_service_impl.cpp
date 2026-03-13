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

#include "sample_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include "common/instance_name.h"
#include "core/framework/request/request_params.h"
#include "core/framework/request/sample_slot.h"
#include "core/util/uuid.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_sample_request_id() {
  return "sample-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

Status build_unimplemented_status() {
  return Status(StatusCode::UNKNOWN,
                "Sample service entry is wired, but selector sampling "
                "logic is not implemented yet");
}

void initialize_response(const std::string& request_id,
                         const std::string& model,
                         proto::SampleResponse* response) {
  CHECK(response != nullptr);
  response->Clear();
  response->set_id(request_id);
  response->set_object("sample_completion");
  response->set_created(
      static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now())));
  response->set_model(model);
}

}  // namespace

namespace sample_service_internal {

Status validate_request(const proto::SampleRequest& request) {
  if (request.model().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "model is required");
  }
  if (request.prompt().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "prompt is required");
  }
  if (!request.has_selector()) {
    return Status(StatusCode::INVALID_ARGUMENT, "selector is required");
  }
  if (request.selector().type() != "literal") {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "selector.type must be literal");
  }
  if (request.selector().value().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "selector.value is required");
  }
  if (request.has_logprobs() &&
      (request.logprobs() < kMinSampleLogprobs ||
       request.logprobs() > kMaxSampleLogprobs)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "logprobs must be between 1 and 5");
  }
  return Status();
}

bool build_request_params(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          RequestParams* request_params) {
  if (request_params == nullptr) {
    return false;
  }

  RequestParams params;
  params.request_id =
      request.has_request_id() ? request.request_id()
                               : generate_sample_request_id();
  params.logprobs = true;
  params.top_logprobs =
      request.has_logprobs() ? request.logprobs() : kDefaultSampleLogprobs;
  params.max_tokens = 1;
  params.n = 1;
  params.best_of = 1;
  params.add_special_tokens = true;
  params.is_sample_request = true;

  if (!build_sample_slots(params.request_id,
                          request.prompt(),
                          request.selector().value(),
                          tokenizer,
                          &params.sample_slots)) {
    return false;
  }

  *request_params = std::move(params);
  return true;
}

bool build_empty_response(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          const std::string& request_id,
                          proto::SampleResponse* response) {
  if (response == nullptr) {
    return false;
  }

  std::vector<int32_t> prompt_tokens;
  if (!tokenizer.encode(request.prompt(), &prompt_tokens, true)) {
    return false;
  }

  initialize_response(request_id, request.model(), response);
  response->mutable_choices();
  auto* usage = response->mutable_usage();
  const int32_t prompt_tokens_count = static_cast<int32_t>(prompt_tokens.size());
  usage->set_prompt_tokens(prompt_tokens_count);
  usage->set_completion_tokens(0);
  usage->set_total_tokens(prompt_tokens_count);
  return true;
}

}  // namespace sample_service_internal

SampleServiceImpl::SampleServiceImpl(LLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

bool SampleServiceImpl::process_request(const proto::SampleRequest& request,
                                        proto::SampleResponse* response,
                                        Status* status) const {
  CHECK(response != nullptr);
  CHECK(status != nullptr);
  response->Clear();

  *status = sample_service_internal::validate_request(request);
  if (!status->ok()) {
    return false;
  }

  if (!models_.contains(request.model())) {
    *status = Status(StatusCode::UNKNOWN, "Model not supported");
    return false;
  }

  RequestParams request_params;
  if (!sample_service_internal::build_request_params(
          request, master_->tokenizer(), &request_params)) {
    *status = Status(StatusCode::UNKNOWN,
                     "Failed to build sample selector runtime mapping");
    return false;
  }

  if (request_params.sample_slots.empty()) {
    if (!sample_service_internal::build_empty_response(
            request, master_->tokenizer(), request_params.request_id, response)) {
      *status = Status(StatusCode::UNKNOWN,
                       "Failed to build sample no-match response");
      return false;
    }
    *status = Status();
    LOG(INFO) << "Sample request fast-returned with no selector matches for "
              << "model " << request.model()
              << ", request_id=" << request_params.request_id;
    return true;
  }

  *status = build_unimplemented_status();
  LOG(WARNING) << "Sample request reached skeleton implementation for model "
               << request.model()
               << ", request_id=" << request_params.request_id
               << ", sample_slots=" << request_params.sample_slots.size();
  return false;
}

void SampleServiceImpl::process_async_impl(std::shared_ptr<SampleCall> call) {
  Status status;
  if (process_request(call->request(), &call->response(), &status)) {
    call->write_and_finish(call->response());
    return;
  }
  call->finish_with_error(status.code(), status.message());
}

}  // namespace xllm
