# `/v1/sample` Demo 契约基线

本文冻结 `/v1/sample` Demo 版的请求/响应/错误语义，作为 ST02-ST12 的唯一实现和测试基线。

## 1. 范围冻结

### 1.1 In Scope

| 项目 | 冻结结论 |
|---|---|
| 接口路径 | `POST /v1/sample` |
| 支持后端 | 仅 `--backend=llm` |
| selector 类型 | 仅 `selector.type=literal` |
| 匹配方式 | 对 `prompt` 做全文顺序匹配 |
| 输出结构 | 复用 completion 风格 `choices[]/logprobs/usage` |
| choice 数量 | 严格等于 `selector.value` 命中数 |
| 顺序语义 | `choices[i].index == sample_id == 第 i 个命中序号` |
| logprobs 默认值 | 服务内常量，默认 `5` |
| logprobs 范围 | `[1, 5]` |

### 1.2 Out of Scope

| 项目 | 说明 |
|---|---|
| 候选集配置 | 不支持请求级或启动级候选配置 |
| 多 token 链式采样 | Demo 版不做 |
| 其他 selector 类型 | 不支持 `regex/token_id` |
| 非 LLM 后端 | 不支持 `vlm/dit/rec` |
| `n` 参数 | 不开放，返回条数完全由命中数决定 |

## 2. 请求契约

| 字段 | 类型 | 必填 | 规则 |
|---|---|---|---|
| `model` | string | 是 | 服务模型名 |
| `prompt` | string | 是 | 完整输入文本 |
| `selector` | object | 是 | 命中规则对象 |
| `selector.type` | string | 是 | 固定为 `literal` |
| `selector.value` | string | 是 | 待匹配字面量，例如 `<emb_0>` |
| `logprobs` | int | 否 | 缺省走服务默认值；显式传入时必须满足 `[1, 5]` |
| `request_id` | string | 否 | 透传到响应与日志 |

补充规则：

1. 请求不接受候选相关字段。
2. 请求不接受 `n` 参数。
3. 字段缺失、类型错误、非法枚举统一走 `INVALID_ARGUMENT`。

## 3. 响应契约

顶层响应固定为 completion 风格：

| 字段 | 类型 | 规则 |
|---|---|---|
| `id` | string | 请求 ID |
| `object` | string | 固定为 `sample_completion` |
| `created` | uint32 | Unix 秒级时间戳 |
| `model` | string | 模型名 |
| `choices` | list | 长度严格等于命中数 |
| `usage` | object | 复用 completion 用量统计 |

`choices[i]` 冻结如下：

| 字段 | 类型 | 规则 |
|---|---|---|
| `index` | int | 从 `0` 连续递增，等于 `sample_id` |
| `text` | string | top-1 token 文本；空 logprobs 时为空串 |
| `logprobs` | object | 与 completion 的 `LogProbs` 结构一致 |
| `finish_reason` | string | 正常为 `selector_match`；异常降级为 `empty_logprobs` |

`choices[i].logprobs` 冻结如下：

| 字段 | 类型 | 规则 |
|---|---|---|
| `tokens` | list<string> | 按 logprob 降序 |
| `token_ids` | list<int> | 与 `tokens` 一一对应 |
| `token_logprobs` | list<float> | 与 `tokens` 一一对应 |

一致性要求：

1. 有效 logprobs 场景下，`choice.text == choice.logprobs.tokens[0]`。
2. `choices` 顺序必须稳定，可直接用 `index` 追踪回 selector 命中位点。
3. 空命中场景不返回伪 choice。

## 4. 行为与错误语义矩阵

| 场景 | HTTP 结果 | xLLM 状态码/语义 | 响应约束 |
|---|---|---|---|
| selector 命中 >= 1 | 200 | 正常 | `choices.size == 命中数` |
| selector 无命中 | 200 | 正常快返 | `choices=[]`，不进入模型采样 |
| 某命中位点无可用 logprobs | 200 | 正常降级 | 对应 `choice.text=\"\"`，3 个 logprobs 数组全空，`finish_reason=\"empty_logprobs\"` |
| `logprobs < 1` 或 `> 5` | 非 200 | `INVALID_ARGUMENT` | 报错信息需指出取值范围 |
| `selector.type != literal` | 非 200 | `INVALID_ARGUMENT` | 报错信息需指出当前仅支持 literal |
| 必填字段缺失/类型错误 | 非 200 | `INVALID_ARGUMENT` | 沿用现有 `finish_with_error` |
| `--backend != llm` | 非 200 | `UNKNOWN` | 消息明确说明当前后端暂不支持 `/v1/sample` |
| 并发限流 | 非 200 | `RESOURCE_EXHAUSTED` | 沿用现有服务限流语义 |
| 服务睡眠 | 非 200 | `UNAVAILABLE` | 沿用现有 sleep 语义 |

## 5. 运行态不变量

后续实现必须保持以下不变量，这些约束直接驱动 ST04-ST09 的数据结构与排序逻辑：

| 不变量 | 约束 |
|---|---|
| 命中建模 | 每个命中位置生成唯一 `SampleSlot` |
| `sample_id` 分配 | 同一请求内按命中出现顺序生成 `0..N-1` |
| 输入注入 | `selected_token_idxes`、`sampling_params`、`sample_idxes` 长度和顺序一致 |
| 输出回填 | 一条模型输出只能回填到一个 `SampleSlot` |
| 聚合排序 | 服务层最终按 `sample_id` 升序组装 `choices[]` |
| 兼容性 | 不能改变 `/v1/completions` 与 `/v1/chat/completions` 现有行为 |

## 6. ST02-ST12 一致性核对

| 任务 | 契约约束 |
|---|---|
| ST02 | `sample.proto` 只暴露 V1 冻结字段，不引入候选集或多 token 扩展字段 |
| ST03 | 路由仅新增 `v1/sample => SampleHttp`，不得影响 completions/chat |
| ST04 | selector 仅做 literal 全文顺序匹配，`sample_id` 从 0 连续编号 |
| ST05 | 完整实现上表错误矩阵与无命中快返 |
| ST06 | 多命中注入必须保序，且不再退化为“只看最后一个 token” |
| ST07 | 允许单 sequence 多输出，空 logprobs 走降级结果 |
| ST08 | `RequestOutput.outputs` 允许多条结果稳定透传 |
| ST09 | `choices[]` 字段名、数量、顺序与 completion 风格完全对齐 |
| ST10 | 文档与日志只补充，不得修改契约本身 |
| ST11 | UT 至少覆盖命中顺序、无命中、空 logprobs、参数越界、链路保序 |
| ST12 | IT/RG 必须覆盖单命中、多命中、混合 batch 和 completions/chat 回归 |

## 7. 基础测试样例

### Case 1: 单命中

```json
{
  "model": "mtp",
  "prompt": "A <emb_0> B",
  "selector": {"type": "literal", "value": "<emb_0>"},
  "logprobs": 2
}
```

预期：

1. 返回 200。
2. `choices.size == 1`。
3. `choices[0].index == 0`。
4. 若有 logprobs，则 `choices[0].text == choices[0].logprobs.tokens[0]`。

### Case 2: 单请求多命中

```json
{
  "model": "mtp",
  "prompt": "<emb_0> x <emb_0> y <emb_0>",
  "selector": {"type": "literal", "value": "<emb_0>"},
  "logprobs": 5
}
```

预期：

1. 返回 200。
2. `choices.size == 3`。
3. `choices[*].index` 依次为 `0,1,2`。
4. 结果顺序与 prompt 中命中顺序一致。

### Case 3: selector 无命中

```json
{
  "model": "mtp",
  "prompt": "plain text",
  "selector": {"type": "literal", "value": "<emb_0>"}
}
```

预期：

1. 返回 200。
2. `choices=[]`。
3. 不进入模型采样路径。

### Case 4: 部分空 logprobs

输入前置条件：命中数为 2，其中一个命中位点下游返回空 logprobs。

预期：

1. 返回 200。
2. 正常位点 `finish_reason=\"selector_match\"`。
3. 异常位点 `text=\"\"`，`tokens/token_ids/token_logprobs` 全空，`finish_reason=\"empty_logprobs\"`。
4. 两个 choice 仍按 `sample_id` 有序返回。

### Case 5: 参数错误

| 请求形态 | 预期 |
|---|---|
| `selector.type=\"regex\"` | `INVALID_ARGUMENT` |
| `logprobs=0` | `INVALID_ARGUMENT` |
| `logprobs=6` | `INVALID_ARGUMENT` |
| 非 llm backend | `UNKNOWN` |

## 8. 验收使用方式

1. ST02-ST09 任何实现评审，优先核对本文件第 2-6 节。
2. ST11/ST12 编写测试时，至少落地第 7 节的 5 类样例。
3. 若后续实现与本文件冲突，应先更新 PRD 和任务板，再修改代码。
