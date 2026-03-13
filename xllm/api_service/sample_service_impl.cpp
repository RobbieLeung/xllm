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

namespace xllm {
namespace {

Status build_unimplemented_status() {
  return Status(StatusCode::UNKNOWN,
                "Sample service entry is wired, but selector sampling "
                "logic is not implemented yet");
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

  *status = build_unimplemented_status();
  LOG(WARNING) << "Sample request reached skeleton implementation for model "
               << request.model();
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
