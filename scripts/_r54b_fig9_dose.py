# -*- coding: utf-8 -*-
"""R54b: Fig 9 caption 视角集修正（180 views）+ §5.10 首句改四能量。"""
import io, sys

APPLY = '--apply' in sys.argv

JOBS = [
    # --- EN: Fig 9 caption 视角集 ---
    ('docs/manuscript/latex/main.tex',
     'Both use the same 45\n'
     'held-out views, so the difference is one of aggregation, not protocol.',
     'Table~\\ref{tab:projdomain} reports 45 held-out test views whereas the curves here pool all 180\n'
     'completed views, which contributes a 0.8--1.2\\% offset; the remainder is one of aggregation.'),
    # --- EN: §5.10 首句 ---
    ('docs/manuscript/latex/main.tex',
     'we applied Poisson resampling to the 3.0~MeV\n'
     'projections at dose fractions $f = 1.0$, $0.5$, $0.25$ and $0.125$, retrained the network at each\n'
     'dose level with three seeds',
     'we applied Poisson resampling to the projections at all four energies and pooled the result as an\n'
     'equal-weight mean across energies, at dose fractions $f = 1.0$, $0.5$, $0.25$ and $0.125$, and\n'
     'retrained the network at each dose level with three seeds'),
    # --- ZH: Fig 9 caption 视角集 ---
    ('docs/manuscript/latex/main_zh.tex',
     '两者都基于同一批 45 个留出视角，故差异来自聚合方式而非评测协议。',
     '表~\\ref{tab:projdomain} 报告的是 45 个留出测试视角，而此处曲线汇总了全部 180 个待补全视角，'
     '二者相差 0.8--1.2\\%；其余差异来自聚合方式。'),
    # --- ZH: §5.10 首句 ---
    ('docs/manuscript/latex/main_zh.tex',
     '我们对 3.0~MeV 的投影施加泊松重采样',
     '我们对全部四个能量的投影施加泊松重采样、并按能量取等权平均汇总'),
]

buf = {}
ok = True
for path, old, new in JOBS:
    if path not in buf:
        buf[path] = io.open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    n = buf[path].count(old)
    print('%-12s count=%d  %s' % (path.split('/')[-1], n, old.split('\n')[0][:52]))
    if n != 1:
        ok = False
        continue
    buf[path] = buf[path].replace(old, new)

if not ok:
    print('FAIL: an anchor is not unique')
    sys.exit(1)

if APPLY:
    for path, s in buf.items():
        io.open(path, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
        print('written', path)
print('OK' if APPLY else 'check passed (use --apply)')
