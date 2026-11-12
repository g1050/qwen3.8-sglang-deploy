# v0.2.0-rc1：Prompt Cache TTL（5m / 1h）

依据 `/data/xkgao/TTL_design/plan1.md` 实现，P0–P3 全部落地。适配层是挂载进容器的 Python 包 `compat/qwen_ttl`，不改镜像；`config/cache.json` 中 `ttl.enabled`、`l3.enabled` 控制开关。

## 对外行为

与 Kimi 官方端点对照校准（`api.moonshot.cn`，`kimi-k3.1-pro`，原始记录在 `results/ttl-v020-rc1/kimi-official/`）：

| 行为 | 官方实测 | 本实现 |
|---|---|---|
| 缓存块粒度 | 256 token（385→256，1228→1024，4718→4608） | 256 token（`QWEN_TTL_BLOCK_TOKENS`） |
| 不传选项 | 写入记在 5m | 同 |
| 命中续期 | +270s、再 +270s 均命中；空闲 330s 未命中 | 同，判定 `now >= expires` |
| 档位锁定 | 5m 请求命中有效 1h 前缀：命中，新增块按 1h 写入；反之按 5m | 同 |
| 锁定链过期后 | 按本次请求档位重写 | 同 |
| 非法 TTL | HTTP 400 | 同 |

设计 5.2 的 S11（"5m 与 1h 不互通"）与官方实际行为不符，以 S6（首次写入锁定档位）为准。

请求：Chat Completions / Responses 用 `prompt_cache_options: {"mode": "implicit", "ttl": "5m" | "1h"}`；Messages 用顶层 `cache_control: {"type": "ephemeral", "ttl": ...}`，消息体内的同名标记被忽略。

响应字段（读取、写入、未缓存三者互斥，之和等于总输入）：

| API | 读取 | 写入 |
|---|---|---|
| Chat Completions | `usage.prompt_tokens_details.cached_tokens`，命中时另有顶层 `usage.cached_tokens`；未命中时两者都不出现 | `usage.prompt_tokens_details.cache_write_tokens` |
| Responses | `usage.input_tokens_details.cached_tokens` | `usage.input_tokens_details.cache_write_tokens` |
| Messages | `usage.cache_read_input_tokens` | `usage.cache_creation_input_tokens`、`usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`；`input_tokens` 不含读取和写入 |

三种 API 都带响应头 `Msh-Usage-Cache-Write-Tokens-5m` / `-1h`（流式响应在首个数据块之前发出，此时请求已被调度，账单已确定）。`cached_tokens_details`（按 device/host/storage 分来源）不再返回。

## 实现要点

| 设计章节 | 落地 |
|---|---|
| 4.1 请求入口 | `http.TTLMiddleware` 解析并校验；TTL 秒数和到达时间戳经 `cache_salt` 载体传给调度器，`Req.__init__` 处还原原始 salt，TTL 不参与缓存命名空间 |
| 4.2 租约表 | `core.LeaseTable`：键为 256 token 块的链式哈希（以 salt / extra_key 为种子），值 `(last_used, expires, ttl)`；内存字典，每个 rank 一份 |
| 2.5 时钟 | 不做每次匹配的跨 rank 广播。唯一时钟是请求到达时间戳的水位；SGLang 已把同一请求列表广播给所有 rank，各 rank 的租约表逐条一致 |
| 4.3 匹配 | `match_prefix`、`_compute_max_prefix_len`、`_prefetch_kvcache` 截到可复用长度（留存上限内） |
| 4.4 续约与写入 | 请求首个 prefill 批次时一次完成：命中块按锁定档位续约，其余 prompt 块建立租约。生成的输出块只登记为"可复用、不计费" |
| 4.5 用量 | 调度器把账单放进 `cached_tokens_details` 带到 HTTP 进程，中间件改写 JSON / SSE 的 usage 并加响应头 |
| 4.6 淘汰 | `TTLStrategy` 包住原生策略：超留存上限最先，其次已过期，其余按原生顺序。L1、L2 共用（已确认 `components/full.py` 的主机侧淘汰用同一策略对象） |
| 4.7 L3 | 关闭原生"每次主机备份都写盘"。L2 淘汰需求每累计 3.2 万 token 扫描一次队尾，把其中"1h 档、租约剩余 > 5 分钟、未写盘"的叶子写入 L3（写入期间节点被锁，淘汰会跳过它），令牌桶限速。只有内存匹配点之后紧接的块**确实写进过 L3 且仍在留存期内**才查盘，查不到不重试；5m 流量完全不碰 L3 |
| 4.8 留存与预算 | `T_keep` 7200 秒。每日写入预算 = DWPD 1 × 盘容量 × 份额 0.5；速率上限和每 rank 配额由 `scripts/cache_profile.py` 按公式算出。超过留存的 L3 文件每 5 分钟清理 |
| 4.10 指标 | `qwen_ttl_tokens_total{kind}`（read / write_5m / write_1h / input / physical / expired_reuse / unbacked_read / l3_written / l3_dropped / l3_lookup）、`qwen_ttl_overhead_seconds_total{part}`（match / admit / storage_gate / tail_scan）、`qwen_ttl_requests_total{tier}`、`qwen_ttl_idle_seconds{kind}`、`qwen_ttl_lease_rows` |

## 与设计稿的差异

- **按租约计费，不取物理长度的较小值**（设计 4.3 第 4 步）。本模型含 Mamba 状态，物理上只能从状态检查点恢复，取较小值会让这个内部细节出现在计费字段里（实测"改动前缀中部"时物理复用 4096、租约有效 7424）。按租约计费后字段与参考模型逐请求一致；租约有效但物理缺失的部分记入 `unbacked_read`，即有效期内被容量淘汰时仍按命中计费、由服务方承担重算。
- **L1/L2 没有主动清除**。超过留存上限的数据匹配不到、淘汰时排最前，但不会被定时释放；L3 文件有定时清除。
- **L3 文件清除按写入时间**（mtime）而非最后使用时间判断，只影响何时删盘，不影响能否被复用。
- **n > 1 或批量请求**的账单取第一个 choice。

## 性能与 L3 调优记录

结果见 [results/README.md](../results/README.md)。1h 档前两次压测不合格，原因和修正：

| 迭代 | 输出吞吐 vs baseline | 原因 | 修正 |
|---|---:|---|---|
| 1 | −24% | 每个 1h 请求都查 L3，查不到重试 8 次，请求卡在调度前；结束后空闲检查报 GPU KV 池账目不一致，调度器退出 | 查盘改为按"实际写入过 L3 的块"精确门控，重试上限 1 |
| 2 | −9.4% | 每次 L2 淘汰都扫描并写入队尾，写入量 55 万 token，引擎写盘线程持续占用调度器进程 | 队尾扫描按淘汰需求摊销；写入预算突发量从 1 小时降到 10 分钟 |
| 3 | −1.0% | — | 采用 |

TTL 记账（匹配、计费、查盘门控、队尾扫描）在 8 并发一小时里合计占调度器约 3.2 秒。

未解决：迭代 1 结束时出现的 `pool memory leak detected`（GPU KV 池 available + evictable 比 total 多 3264 token）没有找到根因；迭代 2、3 和 L3 压力场景中没有再出现。它出现在大量 L3 查盘并发、客户端取消在途请求之后，现在的门控使这条路径很少被走到，但不能证明已消除。

L3 在本机这个负载下收益很小：一小时写入 21 万 token，读回约 1 万 token。按设计 2.7 的容量模型，L2 周转时间约 50 分钟，接近 1 小时，被挤到队尾的条目大多只剩几分钟租约。负载更高、L2 周转更短的部署才会明显受益。

`scripts/ttl/l3_stress.py` 用小 L2 在几分钟内逼出 L3 写入和回读，用来回归 L3 改动（`QWEN_L2_SIZE_GB=6`，结果在 `results/ttl-v020-rc1/l3-stress/`）。

## 运行与验收

```bash
python3 -m unittest discover -s tests && (cd compat && python3 -m unittest qwen_ttl.test_core)
QWEN_TTL_TEST_CLOCK=1 python3 scripts/deploy.py start      # 仅验收：允许请求头注入到达时间
python3 scripts/ttl/acceptance.py --output runtime/acceptance.json
python3 scripts/ttl/kimi_probe.py --help                    # 官方端点 / 本机对照序列
```

`QWEN_TTL_TEST_CLOCK` 只在验收时设置；正式服务不设，时间戳取 HTTP 进程的系统时间。`QWEN_TTL_DEFAULT_SECONDS=3600` 仅供压测使用（AgentX 客户端不带缓存选项，用它让全部请求走 1h）。

验收结果：正式配置和 MTP 压测配置各 51 项全部通过（`results/ttl-v020-rc1/acceptance-*.json`）。

限制沿用设计 4.9：`pp_size` 必须为 1；不支持与显式引擎会话、PD 模式组合。
