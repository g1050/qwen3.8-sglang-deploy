# C8 AgentX 结果

本机工件：`/data/xkgao/qwen3.8-deploy/runtime/agentx-c8-20261112T122320Z`

退出码：`{"benchmark": 0, "aggregate": 0, "validate": 0, "distributions": 0, "server_metrics": 0}`

| 指标 | 数值 |
|---|---:|
| 请求吞吐 req/s | 0.099 |
| 输出吞吐 tok/s | 85.196 |
| 输入吞吐 tok/s | 9197.753 |
| TTFT 平均 s | 29.547 |
| TTFT P50 s | 12.505 |
| TTFT P90 s | 72.029 |
| TTFT P99 s | 271.947 |
| ITL 平均 ms | 75.919 |
| ITL P50 ms | 70.437 |
| ITL P90 ms | 82.881 |
| ITL P99 ms | 139.643 |
| E2E 延迟 平均 s | 91.226 |
| E2E 延迟 P90 s | 166.759 |
| 单用户输出速度 平均 tok/s | 13.842 |
| 输入长度 平均 tok | 92685.475 |
| 输出长度 平均 tok | 863.863 |
| 缓存读取占比 % | 93.848 |

正式请求数：358.0；错误汇总：`[{"error_details": {"type": "ServerDisconnectedError", "message": "ServerDisconnectedError('Server disconnected')", "cause_chain": ["ServerDisconnectedError"]}, "count": 1}]`

时延越低越好，吞吐越高越好。模拟接受率仅用于性能对比，不代表模型输出质量。
