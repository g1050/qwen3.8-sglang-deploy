# Grafana 监控

看板 `Qwen3.8-27B-FP8 推理监控`（uid `qwen38-deploy`）由 `monitoring/build_dashboard.py` 从 Kimi-K3.1 Pro 参考看板（`reference/grafana-kimi-k31-reference.json`）重建，布局一致，共 67 个面板。

```bash
python3 monitoring/build_dashboard.py --install /data/models/monitoring/grafana/dashboards
```

本机 Grafana（`:3000`，文件夹 AI Monitoring）每 30 秒重新加载该目录。地址：`http://192.168.4.35:3000/d/qwen38-deploy`。

与参考看板的差异：

- 选择器改为本机 Prometheus 的 `job="sglang",service="qwen38"`、`model_name="Qwen3.8-27B-FP8"`，指标名为 `sglang:*`。
- GPU 和主机面板改用本机 `nvidia_gpu_*`（`:9400`）与 node-exporter（`:9100`）。
- 去掉收益估算和每请求 TPOT（本版 SGLang 不导出该直方图）。
- HiCache/L3、主机 Mamba 缓存面板保留，L3 关闭时无数据。

Prometheus 采样间隔为 15 秒（参考环境为 2 秒），短时间窗口的速率曲线更平滑。未修改 Prometheus 和 Grafana 的现有配置。
