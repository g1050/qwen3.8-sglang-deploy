# Qwen3.8-27B-FP8 部署管理

**v0.2.0-rc1：** 在 v0.1.0 基础上增加 Prompt Cache TTL（5m / 1h，默认 5m，字段与 Kimi 官方端点对照校准）和按 TTL 门控的 L3，默认全部开启；8 并发性能与 v0.1.0 baseline 持平（−0.6% / −1.0%）。说明见 [docs/ttl-v020.md](docs/ttl-v020.md)，结果见 [results/README.md](results/README.md)。

**v0.1.0：** 本机 8×RTX 5090 上的 SGLang 0.5.20 服务，模型名 `Qwen3.8-27B-FP8`，端口 8000。默认服务为真实解码，**TTL 与 L3 默认关闭**，L2 主机内存缓存按内存自动定容（本机每 rank 47 GB）。压测使用独立的 `performance` 配置（原生 MTP + 模拟接受率），结果见 [results/README.md](results/README.md)，监控见 [docs/monitoring.md](docs/monitoring.md)。

工程参考 `/data/xkgao/kimi3.1-deploy`：只保存配置、脚本、依赖锁、来源记录和小型结果；模型、镜像、数据集、缓存及完整日志均不入 Git。

## 两种配置

| 配置 | 容器 | 用途 | 解码 |
|---|---|---|---|
| `service`（默认） | `qwen38-deploy` | 对外服务，`--restart=unless-stopped` | 真实解码，无投机解码 |
| `performance` | `qwen38-perf` | 仅 InferenceX AgentX 压测 | NEXTN/MTP 3 草稿 token + 1 验证 token，模拟 AL=2.32 |

两者共用 `config/deployment.json` 的基础参数（与迁移前 `qwen38-sglang` 容器一致）：TP8、上下文 262144、`mem-fraction-static 0.90`、最多 140 个运行请求、Triton FP8 GEMM、关闭 CUDA graph 和 custom all-reduce、`NCCL_P2P_DISABLE=1`。

模拟接受率用于排除真实接受率波动对性能的影响：`SGLANG_SIMULATE_ACC_LEN=2.32`、`match-expected`、`real-draft-token`。InferenceX 没有 Qwen3.8-27B 的黄金曲线，按用户选择取 `qwen3.8-flash-next-fp8` MTP 曲线 thinking_on 3 步的值。模拟输出不作为模型正确性结果。

## 缓存分层

`config/cache.json` 默认：L2 按内存自动定容（本机每 rank 47 GB）；TTL 开启（5m / 1h，留存上限 2 小时）；L3 开启，只接收租约有效的 1h 档，配额和写入速率按磁盘寿命预算计算（本机每 rank 约 19 GB、159 token/s）。关闭 TTL 需同时关闭 L3。详见 [docs/ttl-v020.md](docs/ttl-v020.md)。

## 在本机操作

```bash
cd /data/xkgao/qwen3.8-deploy
python3 scripts/deploy.py check            # 权重分片大小与边界
python3 scripts/deploy.py render           # 查看有效启动参数
python3 scripts/deploy.py start && python3 scripts/deploy.py wait
python3 scripts/deploy.py status | logs | stop | remove
python3 scripts/run_sweep.py               # 4、8 并发压测，结束后启动默认服务
python3 -m unittest discover -s tests
```

`start` 在 8000 端口已被其他运行中容器发布时拒绝执行；本工程不会停止非自己启动的容器。`run_sweep.py` 每个并发点使用全新 `performance` 容器（不沿用缓存），预热后测量 3600 秒，无论成功失败都会移除压测容器并启动默认服务。状态在 `runtime/sweep-state.json`，分阶段日志在 `runtime/sweep-*.log`。

## 压测口径

| 项目 | 设置 |
|---|---|
| 上游源码 | InferenceX `0c3f1e24ede23985cf087dae4474e5519a44e37b`，直接调用 `ReplayConfig/replay_argv` 及上游聚合、失败率校验 |
| 客户端 | `agentx-harness==1.0.13`，Python 3.12，`config/client.lock` 带哈希 |
| 数据集 | 完整 `semianalysisai/cc-traces-weka-062126`，revision `23f152f6…`，SHA256 `29b6a19e…`，本机 `/data/models/cc-traces-weka-062126` |
| 上下文筛选 | `--max-context-length 262144`：峰值上下文超过服务上限的轨迹不回放（393 条中 220 条可用） |
| 测量 | 4、8 并发 trajectory lanes，seed 42，每 lane 10 次预热，正式 3600 秒，保留 subagent fan-out |

主机只有 Python 3.10，客户端环境在固定 SGLang 镜像的 Python 3.12 中创建并在该镜像的容器内运行（host 网络）。

## 从头准备

```bash
cp config/host.example.json config/host.json   # 按机器修改路径
python3 scripts/prepare.py client              # 清华 PyPI 镜像，--require-hashes
python3 scripts/prepare.py dataset             # 校验 SHA256，经 hf-mirror 解析元数据后全程离线
```

数据集缺失时 `prepare.py dataset` 通过 `hf_endpoint`（默认 hf-mirror）下载；镜像源在 `host.json` 中可改。

## 回退

迁移前的容器 `qwen38-sglang` 仅被停止、未删除，其配置在 `/data/models/sglang-qwen38/compose.yaml`：

```bash
python3 scripts/deploy.py stop
docker start qwen38-sglang
```

## 版本管理

`git config core.hooksPath .githooks` 后，提交钩子拒绝超过 1 MiB 的暂存文件；`runtime/` 与 `config/host.json` 被忽略。
