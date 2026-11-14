#!/usr/bin/env python3
"""Rebuild the Qwen3.8 dashboard from the Kimi-K3.1 reference layout for the local Prometheus.

Selectors, metric names and exporters are rewritten for this host; panels whose metric the
SGLang 0.5.20 build or this model does not export are dropped rather than left empty.
"""
import argparse
import copy
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
    '': f'## {MODEL} · 本机 8×RTX 5090\n**SGLang 0.5.20 · TP8 · L1 GPU / L2 主机内存 / L3 NVMe · TTL 5m / 1h**　|　'
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


TTL = '{job="sglang",service="qwen38"'
ENG = '{' + ENGINE + f',model_name="{MODEL}",tp_rank="0",pp_rank="0"'
R = '[$__rate_interval]'


def tok(kinds, window=R):
    return f'sum(rate(qwen_ttl_tokens_total{TTL},kind=~"{kinds}"}}{window}))'


PROMPT = tok('read|write_5m|write_1h|input')
ENGINE_ALL = f'sum(rate(sglang:prefill_effective_tokens_total{ENG}}}{R}))'


def engine(mode):
    return f'sum(rate(sglang:prefill_effective_tokens_total{ENG},mode="{mode}"}}{R}))'


def idle(kind, q):
    return f'histogram_quantile({q}, sum by (le) (rate(qwen_ttl_idle_seconds_bucket{TTL},kind="{kind}"}}[5m])))'


# (type, title, width, height, unit, [(expr, legend)], description)
TTL_PANELS = [
    ('row', 'TTL · 计费命中 vs 实际命中', 24, 1, None, [], ''),
    ('stat', '计费命中率', 4, 5, 'percentunit', [(f'{tok("read")} / {PROMPT}', '租约有效 / 输入')],
     '返回给用户的口径：租约有效的块占输入 token 的比例。'),
    ('stat', '实际命中率', 4, 5, 'percentunit', [(f'{tok("physical")} / {PROMPT}', '免算 / 输入')],
     '引擎口径：实际跳过 prefill 的 token 占输入的比例（L1+L2+L3）。'),
    ('stat', '过期内部复用', 4, 5, 'suffix:tok/s', [(tok('expired_reuse'), '实际命中但按写入计费')],
     '租约已过期、数据仍在留存期内被复用：省了计算，对用户计为写入。'),
    ('stat', '计费命中但需重算', 4, 5, 'percentunit', [(f'{tok("unbacked_read")} / {tok("read")}', '占计费命中')],
     '租约有效但数据不在（容量淘汰、状态检查点粒度，或并发请求共享前缀时尚未算完）。按命中计费、由服务方重算。'),
    ('stat', '1h 档请求占比', 4, 5, 'percentunit',
     [(f'sum(rate(qwen_ttl_requests_total{TTL},tier="1h"}}{R})) / sum(rate(qwen_ttl_requests_total{TTL}}}{R}))', '1h / 全部')], ''),
    ('stat', '租约表行数', 4, 5, 'short', [(f'max(qwen_ttl_lease_rows{TTL}}})', '256 token 块')], ''),
    ('timeseries', '命中率 · 计费 vs 实际', 12, 8, 'percentunit',
     [(f'{tok("read")} / {PROMPT}', '计费命中率'), (f'{tok("physical")} / {PROMPT}', '实际命中率'),
      (f'{engine("device_hit")} / {ENGINE_ALL}', '实际 · L1 GPU'), (f'{engine("host_hit")} / {ENGINE_ALL}', '实际 · L2 主机内存'),
      (f'{engine("storage_hit")} / {ENGINE_ALL}', '实际 · L3 磁盘')],
     '计费命中率高于实际：有效租约的数据被淘汰；低于实际：过期数据被内部复用。分层曲线来自引擎原生统计（含输出 token 的续算）。'),
    ('timeseries', '计费 token 构成', 12, 8, 'suffix:tok/s',
     [(tok('read'), '缓存读取'), (tok('write_5m'), '缓存写入 · 5m'), (tok('write_1h'), '缓存写入 · 1h'), (tok('input'), '普通输入 · 不足一块')],
     '四项互斥，之和等于输入 token。'),
    ('timeseries', '计费与实际的差异', 12, 7, 'suffix:tok/s',
     [(tok('expired_reuse'), '过期内部复用 · 实际命中、计费写入'), (tok('unbacked_read'), '计费命中、实际重算')], ''),
    ('timeseries', '前缀再次被使用时的空闲时长', 12, 7, 's',
     [(idle('hit', 0.5), '计费命中 P50'), (idle('hit', 0.95), '计费命中 P95'),
      (idle('expired_reuse', 0.5), '过期复用 P50'), (idle('expired_reuse', 0.95), '过期复用 P95')],
     '用于评估留存上限 T_keep 取多长才划算（设计 4.10）。'),
    ('row', '淘汰 · KV 池与状态池', 24, 1, None, [], ''),
    ('timeseries', 'GPU 淘汰 · 按池和租约状态', 8, 7, 'short',
     [(f'sum by (pool, lease) (rate(qwen_ttl_evictions_total{TTL},layer="device"}}{R}))', '{{pool}} · {{lease}}')],
     'kv 单位为 token/s，mamba 为状态槽/s。valid 表示被淘汰时租约仍有效。只有 mamba 在动而 kv 不动，说明先见底的是状态池。'),
    ('timeseries', '主机（L2）淘汰 · 按池和租约状态', 8, 7, 'short',
     [(f'sum by (pool, lease) (rate(qwen_ttl_evictions_total{TTL},layer="host"}}{R}))', '{{pool}} · {{lease}}')],
     'kv 单位为 token/s，mamba 为状态槽/s。'),
    ('timeseries', '主机状态槽占用', 8, 7, 'short',
     [(f'max(qwen_ttl_host_state_slots{TTL},state="used"}})', '已用'),
      (f'max(qwen_ttl_host_state_slots{TTL},state="total"}})', '总量')],
     '主机 Mamba 状态池（引擎不导出，由 TTL 适配层采样）。GPU 状态槽见上方 Mamba 面板。'),
    ('row', 'L3 · 本地 NVMe（仅 1h 档）', 24, 1, None, [], ''),
    ('timeseries', 'L3 写入与丢弃', 8, 7, 'suffix:tok/s',
     [(tok('l3_written'), '写入'), (tok('l3_dropped'), '丢弃 · 写入预算用完'),
      (tok('l3_skipped_.*'), '跳过 · 无状态 / 非 1h / 将到期 / 无主机副本')],
     'L2 淘汰队尾里租约有效的 1h 条目。丢弃表示寿命预算不足。'),
    ('timeseries', 'L3 查盘与读回', 8, 7, 'short',
     [(f'sum(increase(qwen_ttl_tokens_total{TTL},kind="l3_lookup"}}{R}))', '查盘次数'),
      (f'sum(increase(sglang:prefill_effective_tokens_total{ENG},mode="storage_hit"}}{R}))', '读回 token')],
     '只有内存匹配点之后的块确实写进过 L3 才查盘。'),
    ('timeseries', 'L3 累计写入量 · 所选时段', 8, 7, 'short',
     [(f'sum(increase(qwen_ttl_tokens_total{TTL},kind="l3_written"}}[$__range]))', '写入 token'),
      (f'sum(increase(qwen_ttl_tokens_total{TTL},kind="l3_dropped"}}[$__range]))', '丢弃 token')],
     '每 token 约 170–240 KB（8 个 rank 合计，含 Mamba 状态）。'),
    ('timeseries', 'TTL 记账开销 · 调度器 CPU', 12, 7, 'suffix:cores',
     [(f'sum by (part) (rate(qwen_ttl_overhead_seconds_total{TTL}}}{R}))', '{{part}}')],
     'match 匹配截短，admit 续约与计费，storage_gate 查盘门控，tail_scan L2 队尾扫描。'),
    ('timeseries', 'L3 预取未兑现 / 推迟', 12, 7, 'suffix:tok/s',
     [(f'sum(rate(sglang:storage_prefetch_unfulfilled_tokens_total{ENG}}}{R}))', '未兑现'),
      (f'sum(rate(sglang:storage_prefetch_deferred_tokens_total{ENG}}}{R}))', '推迟')], '引擎原生指标。'),
]


def ttl_panels(samples, top):
    """Lay the TTL rows out from grid row `top`, cloning the reference board's panel styling."""
    out, x, y, line = [], 0, top, 0
    for kind, title, width, height, unit, targets, description in TTL_PANELS:
        if x + width > 24:
            x, y, line = 0, y + line, 0
        panel = copy.deepcopy(samples[kind])
        panel.update(title=title, description=description, gridPos={'h': height, 'w': width, 'x': x, 'y': y})
        panel.pop('panels', None)
        if kind == 'row':
            panel['collapsed'] = False
        else:
            panel['fieldConfig']['defaults']['unit'] = unit
            panel['fieldConfig']['overrides'] = []
            first = panel['targets'][0]
            panel['targets'] = [{**first, 'expr': query, 'legendFormat': legend, 'refId': chr(65 + i)}
                                for i, (query, legend) in enumerate(targets)]
        out.append(panel)
        x, line = x + width, max(line, height)
    return out, y + line - top


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
        panels.append(panel)
    samples = {}
    for panel in panels:
        samples.setdefault(panel['type'], panel)
    # Insert right below the core-metrics block; everything after it moves down.
    top = next(p['gridPos']['y'] for p in panels if p['type'] == 'row' and p['title'] == '容量与并发')
    added, height = ttl_panels(samples, top)
    for panel in panels:
        if panel['gridPos']['y'] >= top:
            panel['gridPos']['y'] += height
    panels = sorted(panels + added, key=lambda p: (p['gridPos']['y'], p['gridPos']['x']))
    for index, panel in enumerate(panels):
        panel['id'] = index + 1
    board.update(panels=panels, id=None, uid='qwen38-deploy', version=3, title=f'{MODEL} 推理监控',
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
