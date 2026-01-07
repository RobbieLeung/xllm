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

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_qwen3_decoder_layer_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"

namespace xllm {

// EAGLE-3 specific decoder layer that accepts embeds and hidden_states
// separately, applies layernorms, then concatenates them
class QWen3Eagle3DecoderLayerImpl : public torch::nn::Module {
 public:
  QWen3Eagle3DecoderLayerImpl(const ModelContext& context,
                              const int32_t layer_id = 0)
      : layer_id_(layer_id) {
    CHECK(layer_id_ >= 0) << "layer_id must be >= 0, but got " << layer_id_;
    // register submodules
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuQwen3DecoderLayer(context));

    input_layernorm_ =
        register_module("input_layernorm", layer::NpuRMSNorm(context));

    hidden_norm_ = register_module("hidden_norm", layer::NpuRMSNorm(context));
  }

  // Forward with separate embeds and hidden_states (matches Python
  // implementation)
  virtual torch::Tensor forward(torch::Tensor& embeds,
                                torch::Tensor& hidden_states,
                                torch::Tensor& cos_pos,
                                torch::Tensor& sin_pos,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                aclrtEvent* event,
                                std::atomic<bool>* event_flag) {
    torch::Tensor normed_embeds = input_layernorm_(embeds, 0);
    torch::Tensor normed_hidden = hidden_norm_(hidden_states, 0);

    // Concatenate embeds and hidden_states for the QKV projection
    // This creates 2x hidden_size input: [embeds, hidden_states]
    torch::Tensor layer_input =
        torch::cat({normed_embeds, normed_hidden}, /*dim=*/-1);

    return decoder_layer_(layer_input,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag,
                          layer_id_);
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    hidden_norm_->verify_loaded_weights(prefix + "hidden_norm.");
  }
  virtual void merge_loaded_weights() {
    decoder_layer_->merge_loaded_weights();
    input_layernorm_->merge_loaded_weights();
    hidden_norm_->merge_loaded_weights();
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(
        state_dict.get_dict_with_prefix("midlayer."));
    input_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("input_layernorm."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
  }

 private:
  layer::NpuQwen3DecoderLayer decoder_layer_{nullptr};
  // layernorm for embeds
  layer::NpuRMSNorm input_layernorm_{nullptr};
  // layernorm for hidden_states
  layer::NpuRMSNorm hidden_norm_{nullptr};
  int32_t layer_id_;
};
TORCH_MODULE(QWen3Eagle3DecoderLayer);

class QWen3Eagle3ModelImpl : public torch::nn::Module {
 public:
  QWen3Eagle3ModelImpl(const std::string& model_type,
                       const ModelContext& context)
      : model_type_(model_type) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    mrope_section_ = model_args.rope_scaling_mrope_section();

    is_mrope_enabled_ = !mrope_section_.empty();

    // Word embedding
    embed_tokens_ =
        register_module("embed_tokens", layer::NpuWordEmbedding(context));

    // Position embedding
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args.head_dim(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    // Final norm
    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    // EAGLE-3 fusion layer: 3 * target_hidden_size -> hidden_size
    // Get target_hidden_size from model_args, default to hidden_size
    target_hidden_size_ = model_args.target_hidden_size();
    if (target_hidden_size_ == 0) {
      target_hidden_size_ = model_args.hidden_size();
    }

    // fc layer for fusion: 3 * target_hidden_size -> hidden_size
    fc_ = register_module("fc", layer::NpuColumnParallelLinear(context));

    // EAGLE-3 has only 1 layer
    decoder_ = register_module("midlayer", QWen3Eagle3DecoderLayer(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);

    // Handle empty tokens case for dp
    if (dp_size_ > 1 && tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }

    // Get embeddings from tokens
    torch::Tensor embeds = embed_tokens_(tokens, 0);

    // Get hidden_states from input_params.input_embedding
    // In EAGLE-3, hidden_states comes from verifier layers (3 layers
    // concatenated)
    torch::Tensor hidden_states = input_params.input_embedding;
    if (!hidden_states.defined() || hidden_states.size(0) == 0) {
      LOG(WARNING) << "hnorm use embedding from tokens.";
      hidden_states = embeds;
    }

    // Apply fusion if hidden_states dimension doesn't match embeds
    // hidden_states shape: [B*L, 3*target_hidden_size] or [B*L, hidden_size]
    if (hidden_states.size(-1) != embeds.size(-1)) {
      hidden_states = fc_(hidden_states, 0);
    }

    // Compute positional embeddings
    torch::Tensor target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    // Handle mrope case
    if (is_mrope_enabled_ && positions.dim() == 2) {
      auto apply = [this](torch::Tensor x) {
        auto sections = mrope_section_;
        sections.insert(sections.end(), sections.begin(), sections.end());

        auto vec = x.split(sections, -1);
        std::vector<torch::Tensor> selects;
        selects.reserve(vec.size());

        for (int64_t i = 0; i < vec.size(); ++i) {
          auto m = vec[i];
          selects.push_back(m[i % mrope_section_.size()]);
        }
        return torch::cat(selects, -1);
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    // Generate attention mask
    torch::Tensor attn_mask;
    if (!input_params.batch_forward_type.is_decode()) {
      if (FLAGS_enable_chunked_prefill) {
        int num_sequences = input_params.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask =
                attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                           input_params.kv_seq_lens_vec[j],
                                           input_params.kv_max_seq_len,
                                           cos_pos.dtype().toScalarType(),
                                           cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    // EAGLE-3 has only 1 layer
    aclrtEvent* event{nullptr};
    std::atomic<bool>* event_flag{nullptr};

    if (input_params.layer_synchronizer != nullptr) {
      event = input_params.layer_synchronizer->get_event(0);
      event_flag = input_params.layer_synchronizer->get_event_flag(0);
    }
    if (!input_params.synchronize_layer(0)) {
      return ModelOutput();
    }

    // The decoder layer applies input_layernorm to embeds and hidden_norm to
    // hidden_states, then concatenates them to create 2x hidden_size input for
    // QKV projection.
    torch::Tensor h_out = decoder_(embeds,
                                   hidden_states,
                                   cos_pos,
                                   sin_pos,
                                   attn_mask,
                                   kv_caches[0],
                                   input_params_new,
                                   event,
                                   event_flag);

    // Apply final norm with residual
    // The norm returns hidden_states_to_logits
    torch::Tensor hidden_states_to_logits = norm_(h_out, 0);

    // For draft decode, we capture the hidden state before norm as
    // aux_hidden_states This is used for speculative decoding to pass hidden
    // states to next step
    return ModelOutput(hidden_states_to_logits,
                       /*residual=*/torch::Tensor(),
                       /*aux_hidden_states=*/h_out);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    // Note: embed_tokens is shared from target model, not loaded here
    // Only load if the weight exists in state_dict
    torch::Tensor embed_weight = state_dict.get_tensor("embed_tokens.weight");
    if (embed_weight.defined()) {
      embed_tokens_->load_state_dict(
          state_dict.get_dict_with_prefix("embed_tokens."));
    }

    // Load EAGLE-3 specific modules
    // fc: (hidden_size, 3*target_hidden_size) fusion layer
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));

    // Load midlayer (single decoder layer) weights
    // This includes: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
    // post_attention_layernorm, input_layernorm, hidden_norm
    decoder_->load_state_dict(state_dict.get_dict_with_prefix("midlayer."));

    // Load final norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  // Load d2t and t2d from root level state_dict (without prefix)
  virtual void load_token_mappings(const StateDict& state_dict) {
    // Handle d2t (draft-to-target) token mapping
    // d2t stores diffs between draft id and target id
    // hot_token_id = d2t + arange(d2t.size(0))
    torch::Tensor d2t_tensor = state_dict.get_tensor("d2t");
    if (!d2t_tensor.defined()) {
      // Try with .weight suffix
      d2t_tensor = state_dict.get_tensor("d2t.weight");
    }
    if (d2t_tensor.defined()) {
      // hot_token_id = d2t + arange(d2t.size(0))
      hot_token_id_ =
          d2t_tensor + torch::arange(d2t_tensor.size(0)).to(d2t_tensor);
      LOG(INFO) << "Loaded d2t tensor, hot_token_id size: "
                << hot_token_id_.size(0);
    }

    // Handle t2d (target-to-draft) token mapping
    // t2d maps target token ids to draft token ids
    torch::Tensor t2d_tensor = state_dict.get_tensor("t2d");
    if (!t2d_tensor.defined()) {
      // Try with .weight suffix
      t2d_tensor = state_dict.get_tensor("t2d.weight");
    }
    if (t2d_tensor.defined()) {
      t2d_ = t2d_tensor;
      LOG(INFO) << "Loaded t2d tensor, size: " << t2d_.size(0);
    }
  }

  virtual void verify_loaded_weights(const std::string& prefix) const {
    // embed_tokens may be shared from target model
    // Only verify if it was loaded
    // embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");

    fc_->verify_loaded_weights(prefix + "fc.");

    // Verify midlayer (decoder layer) - includes input_layernorm and
    // hidden_norm
    decoder_->verify_loaded_weights(prefix + "midlayer.");

    norm_->verify_loaded_weights(prefix + "norm.");
  }

  virtual void merge_loaded_weights() {
    embed_tokens_->merge_loaded_weights();
    fc_->merge_loaded_weights();

    // Merge midlayer (decoder layer) weights - includes input_layernorm and
    // hidden_norm
    decoder_->merge_loaded_weights();

    norm_->merge_loaded_weights();
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return embed_tokens_;
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    embed_tokens_ = npu_word_embedding;
  }

  // Get hot_token_id for draft-to-target token mapping
  torch::Tensor get_hot_token_id() const { return hot_token_id_; }

  // Get t2d for target-to-draft token mapping
  torch::Tensor get_t2d() const { return t2d_; }

 protected:
  std::string model_type_;

  int32_t dp_rank_ = 0;
  int32_t dp_size_ = 1;
  int32_t dp_local_tp_size_ = 1;

  // Rotary embedding
  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  layer::AttentionMask attn_mask_;

  std::vector<int64_t> mrope_section_;
  bool is_mrope_enabled_ = false;

  // Word embedding
  layer::NpuWordEmbedding embed_tokens_{nullptr};

  // EAGLE-3 specific modules
  layer::NpuColumnParallelLinear fc_{nullptr};  // fusion layer
  layer::NpuRMSNorm norm_{nullptr};             // final norm

  // Decoder
  QWen3Eagle3DecoderLayer decoder_{nullptr};

  // EAGLE-3 target hidden size
  int64_t target_hidden_size_ = 0;

  // d2t (draft-to-target) token mapping
  // hot_token_id = d2t + arange(d2t.size(0))
  torch::Tensor hot_token_id_;

  // t2d (target-to-draft) token mapping
  // Maps target token ids to draft token ids
  torch::Tensor t2d_;

  bool layer_forward_interrupted_ = false;
};
TORCH_MODULE(QWen3Eagle3Model);

class QWen3Eagle3ForCausalLMImpl : public torch::nn::Module {
 public:
  QWen3Eagle3ForCausalLMImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    tie_word_embeddings_ = model_args.tie_word_embeddings();

    // register submodules
    model_ =
        register_module("model", QWen3Eagle3Model("qwen3_eagle3", context));

    npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));

    // Check if we need to load lm_head from target model
    load_lm_head_from_target_ = false;
    if (!tie_word_embeddings_) {
      int64_t draft_vocab_size = model_args.draft_vocab_size();
      if (draft_vocab_size == 0) {
        load_lm_head_from_target_ = true;
      }
    }

    capture_aux_hidden_states_ = true;
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    return npu_lm_head_(hidden_states, seleted_idxes, 0);
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      // Load model weights (fc, midlayer, norm)

      model_->load_state_dict(state_dict->get_dict_with_prefix(""));

      // Load d2t and t2d from root level (always without prefix)
      model_->load_token_mappings(*state_dict);

      // Load lm_head
      if (tie_word_embeddings_) {
        // Share weights with embed_tokens
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
      } else if (!load_lm_head_from_target_) {
        // Load from lm_head.weight in checkpoint
        npu_lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix("lm_head."));
      }
      // If load_lm_head_from_target_ is true, lm_head will be loaded from
      // target model externally
    }

    // verify
    model_->verify_loaded_weights(prefix);
    if (!load_lm_head_from_target_) {
      npu_lm_head_->verify_loaded_weights("lm_head.");
    }

    model_->merge_loaded_weights();
    if (!load_lm_head_from_target_) {
      npu_lm_head_->merge_loaded_weights();
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  virtual void set_npu_lm_head(layer::NpuLmHead& head) {
    if (load_lm_head_from_target_) {
      npu_lm_head_ = head;
    }
  }

  virtual layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  virtual void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

  // Get hot_token_id for draft-to-target token mapping
  torch::Tensor get_hot_token_id() const { return model_->get_hot_token_id(); }

  // Get t2d for target-to-draft token mapping
  torch::Tensor get_t2d() const { return model_->get_t2d(); }

  bool should_load_lm_head_from_target() const {
    return load_lm_head_from_target_;
  }

 protected:
  // parameter members, must be registered
  QWen3Eagle3Model model_{nullptr};
  int device_id_ = 0;
  bool tie_word_embeddings_{false};
  bool load_lm_head_from_target_{false};
  bool capture_aux_hidden_states_{true};
  layer::NpuLmHead npu_lm_head_{nullptr};
};
TORCH_MODULE(QWen3Eagle3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_eagle3, QWen3Eagle3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3_eagle3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_eagle3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // EAGLE-3 specific parameters
  LOAD_ARG_OR(draft_vocab_size, "draft_vocab_size", 0);
  LOAD_ARG_OR(target_vocab_size, "target_vocab_size", 0);
  LOAD_ARG_OR(target_hidden_size, "target_hidden_size", 0);
  LOAD_ARG_OR(norm_before_residual, "norm_before_residual", false);

  // For qwen3/2.5 model < 7B, tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  SET_ARG(skip_input_layernorm, true);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
