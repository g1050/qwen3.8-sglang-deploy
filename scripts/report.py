#!/usr/bin/env python3
"""Summarize one AgentX run from AIPerf's own export (and the upstream aggregate when present)."""
import argparse
import json
from pathlib import Path
import sys

from deploy import RUNTIME

ROWS = [
    ('请求吞吐 req/s', 'request_throughput', 'avg', 1),
    ('输出吞吐 tok/s', 'output_token_throughput', 'avg', 1),
    ('TTFT 平均 s', 'time_to_first_token', 'avg', 1e-3),
    ('TTFT P50 s', 'time_to_first_token', 'p50', 1e-3),
    ('TTFT P90 s', 'time_to_first_token', 'p90', 1e-3),
    ('TTFT P99 s', 'time_to_first_token', 'p99', 1e-3),
    ('ITL 平均 ms', 'inter_token_latency', 'avg', 1),
    ('ITL P50 ms', 'inter_token_latency', 'p50', 1),
    ('ITL P90 ms', 'inter_token_latency', 'p90', 1),
    ('ITL P99 ms', 'inter_token_latency', 'p99', 1),
    ('E2E 延迟 平均 s', 'request_latency', 'avg', 1e-3),
    ('E2E 延迟 P90 s', 'request_latency', 'p90', 1e-3),
    ('单用户输出速度 平均 tok/s', 'output_token_throughput_per_user', 'avg', 1),
    ('输入长度 平均 tok', 'input_sequence_length', 'avg', 1),
    ('输出长度 平均 tok', 'output_sequence_length', 'avg', 1),
    ('缓存读取占比 %', 'overall_usage_prompt_cache_read_pct', 'avg', 1),
]


def summarize(run):
    export = json.loads((run / 'aiperf_artifacts/profile_export_aiperf.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    checks = json.loads((run / 'exit-codes.json').read_text())
    summary = {'run': run.name, 'concurrency': manifest['concurrency'], 'duration': manifest['duration'], 'exit_codes': checks}
    for label, key, stat, scale in ROWS:
        metric = export.get(key) or {}
        value = metric.get(stat)
        summary[label] = None if value is None else value * scale
        summary.setdefault('units', {})[label] = metric.get('unit')
    for key in ('request_count', 'total_usage_prompt_tokens', 'total_usage_prompt_cache_read_tokens',
                'total_usage_completion_tokens', 'total_output_tokens', 'benchmark_duration'):
        summary[key] = (export.get(key) or {}).get('avg')
    if summary['total_usage_prompt_tokens'] and summary['benchmark_duration']:
        summary['输入吞吐 tok/s'] = summary['total_usage_prompt_tokens'] / summary['benchmark_duration']
    summary['error_summary'] = export.get('error_summary')
    summary['was_cancelled'] = export.get('was_cancelled')
    aggregate = run / 'aggregate.json'
    if aggregate.is_file():
        summary['upstream_aggregate'] = {k: v for k, v in json.loads(aggregate.read_text()).items()
                                         if k in ('request_metrics', 'request_accounting')}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path)
    args = parser.parse_args()
    run = args.run or Path((RUNTIME / 'latest-run.txt').read_text().strip())
    summary = summarize(run)
    lines = [f"# C{summary['concurrency']} AgentX 结果", '', f'本机工件：`{run}`', '',
             f"退出码：`{json.dumps(summary['exit_codes'])}`", '', '| 指标 | 数值 |', '|---|---:|']
    for label in [row[0] for row in ROWS[:2]] + ['输入吞吐 tok/s'] + [row[0] for row in ROWS[2:]]:
        value = summary.get(label)
        lines.append(f'| {label} | {value:.3f} |' if isinstance(value, (int, float)) else f'| {label} | 无 |')
    lines += ['', f"正式请求数：{summary['request_count']}；错误汇总：`{json.dumps(summary['error_summary'], ensure_ascii=False)}`", '',
              '时延越低越好，吞吐越高越好。模拟接受率仅用于性能对比，不代表模型输出质量。']
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print('\n'.join(lines))
    if any(summary['exit_codes'].values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
