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

bool build_sample_request_params(const proto::SampleRequest& request,
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
  params.top_logprobs = request.has_logprobs() ? request.logprobs() : 5;
  params.max_tokens = 1;
  params.n = 1;
  params.best_of = 1;
  params.add_special_tokens = true;
  params.is_sample_request = true;

  if (request.has_selector() && request.selector().type() == "literal") {
    if (!build_sample_slots(params.request_id,
                            request.prompt(),
                            request.selector().value(),
                            tokenizer,
                            &params.sample_slots)) {
      return false;
    }
  }

  *request_params = std::move(params);
  return true;
}

}  // namespace

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

  if (!models_.contains(request.model())) {
    *status = Status(StatusCode::UNKNOWN, "Model not supported");
    return false;
  }

  RequestParams request_params;
  if (!build_sample_request_params(
          request, master_->tokenizer(), &request_params)) {
    *status = Status(StatusCode::UNKNOWN,
                     "Failed to build sample selector runtime mapping");
    return false;
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
