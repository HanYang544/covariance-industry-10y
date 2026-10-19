"""Publish only the 11 statistical baselines; no training or selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('LW-20', 'LW-60', 'LW-120', 'LW-252', 'EWMA-0.94', 'EWMA-0.97',
           'IEWMA-21-63', 'IEWMA-63-125', 'IEWMA-125-250', 'IEWMA-250-500', 'IEWMA-500-1000')
SEGMENTS = ('2021_2025',)


def select_baselines(rows):
    selected = [dict(method=row['method'], segment='2021_2025', frobenius=row['test_frobenius'])
                for row in rows if row['method'] in METHODS]
    keys = [(row['segment'], row['method']) for row in selected]
    expected = {(segment, method) for segment in SEGMENTS for method in METHODS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Need exactly one score for each of 11 baselines in both periods.')
    return sorted(selected, key=lambda row: (SEGMENTS.index(row['segment']), METHODS.index(row['method'])))


def main():
    source = ROOT / 'outputs/dynamic_holdout/test_results.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = select_baselines(csv.DictReader(stream))
    output = ROOT / 'outputs/baseline_catalog'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'segment', 'frobenius'])
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(method_count=len(METHODS), methods=METHODS, source=str(source.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_stage='dynamic_holdout', fresh_training=False, controls_included=False,
                    loss_branch='Native statistical estimators; no supervised baseline fitting',
                    target='future 20-day average uncentered daily second moment',
                    historical_results_preserved=False, ranking_used_to_filter=False)
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# 统计基线结果', '', '当前只列 LW、EWMA、IEWMA 三类、11 个设置。', '',
             '这份表从已完成实验逐项提取，不是新的训练。仅保留统计估计方法；项目融合基线已移除。具体初始化与稳定化设置见方法说明。', '',
             '[方法与简易推导](../../docs/baselines.md) · [结果 CSV](summary.csv) · [来源与清单](manifest.json)', '',
             '同一未来20日未中心化二阶矩标签、同一预测日期；Frobenius越小越好。', '']
    for segment, label in [('2021_2025', '2021—2025：62 个预测区间')]:
        lines += ['## ' + label, '', '| 方法 | Frobenius |', '| --- | --- |']
        lines += [f"| {row['method']} | {float(row['frobenius']):.9f} |"
                  for row in rows if row['segment'] == segment]
        lines.append('')
    lines += ['该测试时期此前已被观察，不是新的独立验证。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Published {len(METHODS)} baselines, {len(rows)} score rows; no controls or training: {output}')


if __name__ == '__main__':
    main()
