#!/usr/bin/env python3
"""Rebuild the Qwen3.8 dashboard from the Kimi-K3.1 reference layout for the local Prometheus.

Selectors, metric names and exporters are rewritten for this host; panels whose metric the
SGLang 0.5.20 build or this model does not export are dropped rather than left empty.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config/deployment.json').read_text())
MODEL = CFG['served_model_name']
ENGINE = 'job="sglang",service="qwen38"'
DROP_ROWS = {'收益估算 · Kimi K3.1 Pro（人民币）', '请求 TPOT 与有效前缀缓存'}
DROP_PANELS = {'每请求 TPOT'}
EXPORTERS = [
    (r'deepseek_gpu_utilization_ratio\{[^}]*\}', '(nvidia_gpu_utilization_percent{job="nvidia-gpu"} / 100)'),
    (r'deepseek_gpu_memory_used_bytes\{[^}]*\}', '(nvidia_gpu_memory_used_mib{job="nvidia-gpu"} * 1048576)'),
    (r'deepseek_gpu_memory_total_bytes\{[^}]*\}', '(nvidia_gpu_memory_total_mib{job="nvidia-gpu"} * 1048576)'),
    (r'deepseek_gpu_power_watts\{[^}]*\}', 'nvidia_gpu_power_draw_watts{job="nvidia-gpu"}'),
    (r'deepseek_gpu_temperature_celsius\{[^}]*\}', 'nvidia_gpu_temperature_celsius{job="nvidia-gpu"}'),
    (r'deepseek_host_memory_total_bytes\{[^}]*\}', 'node_memory_MemTotal_bytes{job="node"}'),
    (r'deepseek_host_memory_available_bytes\{[^}]*\}', 'node_memory_MemAvailable_bytes{job="node"}'),
    (r'deepseek_host_load(1|5|15)\{[^}]*\}', r'node_load\1{job="node"}'),
    (r'\{job="deepseek-v4-gpu",instance="127\.0\.0\.1:9835"\}', '{job="nvidia-gpu"}'),
    (r'\{job="deepseek-v4-prometheus",instance="127\.0\.0\.1:9091"\}', '{job="prometheus"}'),
]
TEXT = {
    '': f'## {MODEL} · 本机 8×RTX 5090\n**SGLang 0.5.20 · TP8 · 原生 GPU 前缀缓存 · TTL / L3 默认关闭**　|　'
        f"服务 `:{CFG['port']}`　|　采样 15 秒，页面刷新 5 秒  \n业务空闲时延迟/命中率可能无样本；"
        'HTTP 统计来自引擎。压测期间端口由模拟接受率的 MTP 容器提供，投机解码面板随之出现数据。',
    'HTTP 延迟 · 采集范围': '当前 `/metrics` **未导出 HTTP duration 直方图**，因此不显示 HTTP P95。\n\n'
                     '首 token、ITL、请求总延迟见上方引擎面板。',
    '采集口径与配置': f"**数据来源**：本机 Prometheus `:9090`（job `sglang` → `:{CFG['port']}/metrics`），"
               'GPU exporter `:9400`，node-exporter `:9100`。默认最近 6 小时，5 秒刷新。\n\n'
               f'**模型筛选**：`model_name="{MODEL}"`。投机解码状态以实时指标为准；无样本保留为空。\n\n'
               '逻辑 token 选 TP0。HiCache/L3 面板仅在 `config/cache.json` 开启 L3 后有数据。\n\n'
               '面板由 `/data/xkgao/qwen3.8-deploy/monitoring/build_dashboard.py` 从参考看板重建。',
}


def expr(text):
    text = re.sub(r'\{job="kimi-k3-vllm",instance="127\.0\.0\.1:30000",model_name=~"[^"]*"',
                  '{' + ENGINE + f',model_name="{MODEL}"', text)
    text = text.replace('{job="kimi-k3-vllm",instance="127.0.0.1:30000"', '{' + ENGINE)
    for pattern, replacement in EXPORTERS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\bsglang_', 'sglang:', text)
    if 'kimi' in text.lower() or 'deepseek' in text.lower():
        raise ValueError(f'Unconverted expression: {text}')
    return text


def label(text):
    for old, new in (('Kimi-K3.1 Pro', MODEL), ('Kimi K3.1 Pro', MODEL), ('8×B300', '8×RTX 5090'),
                     (':30000', f":{CFG['port']}")):
        text = text.replace(old, new)
    return text


def build(reference):
    board = json.loads(reference.read_text())
    panels, skipping = [], False
    for panel in board['panels']:
        if panel['type'] == 'row':
            skipping = panel['title'] in DROP_ROWS
        if skipping or panel.get('title') in DROP_PANELS:
            continue
        panel['title'] = label(panel.get('title', ''))
        if panel['type'] == 'text':
            panel['options']['content'] = TEXT[panel['title']]
        for target in panel.get('targets', []):
            target['expr'] = expr(target['expr'])
            target['legendFormat'] = label(target.get('legendFormat', ''))
        panel['id'] = len(panels) + 1
        panels.append(panel)
    board.update(panels=panels, id=None, uid='qwen38-deploy', version=1, title=f'{MODEL} 推理监控',
                 tags=['qwen3.8', 'sglang'], links=[],
                 description=f"本机 {MODEL} / SGLang / 8×RTX 5090；:{CFG['port']} 服务指标，复用现有 Prometheus、GPU exporter 和 node-exporter。")
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'reference/grafana-kimi-k31-reference.json')
    parser.add_argument('--install', type=Path, help='Grafana provisioned dashboards directory to copy into')
    args = parser.parse_args()
    output = ROOT / 'monitoring/grafana/qwen38-deploy.json'
    output.write_text(json.dumps(build(args.reference), ensure_ascii=False, indent=1) + '\n')
    print(output)
    if args.install:
        shutil.copy2(output, args.install / output.name)
        print(args.install / output.name)


if __name__ == '__main__':
    main()
