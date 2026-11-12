# C8 AgentX 结果

本机工件：`/data/xkgao/qwen3.8-deploy/runtime/agentx-c8-20261112T191818Z`

退出码：`{"benchmark": 0, "aggregate": 0, "validate": 0, "distributions": 0, "server_metrics": 0}`

| 指标 | 数值 |
|---|---:|
| 请求吞吐 req/s | 0.098 |
| 输出吞吐 tok/s | 84.348 |
| 输入吞吐 tok/s | 9148.221 |
| TTFT 平均 s | 30.099 |
| TTFT P50 s | 13.804 |
| TTFT P90 s | 77.568 |
| TTFT P99 s | 262.378 |
| ITL 平均 ms | 75.396 |
| ITL P50 ms | 71.106 |
| ITL P90 ms | 87.123 |
| ITL P99 ms | 132.230 |
| E2E 延迟 平均 s | 93.014 |
| E2E 延迟 P90 s | 172.302 |
| 单用户输出速度 平均 tok/s | 13.577 |
| 输入长度 平均 tok | 93177.860 |
| 输出长度 平均 tok | 860.062 |
| 缓存读取占比 % | 94.437 |

正式请求数：356.0；错误汇总：`[{"error_details": {"type": "ServerDisconnectedError", "message": "ServerDisconnectedError('Server disconnected')", "cause_chain": ["ServerDisconnectedError"]}, "count": 1}]`

时延越低越好，吞吐越高越好。模拟接受率仅用于性能对比，不代表模型输出质量。
