# -*- coding: utf-8 -*-
"""R53c: 表注补池化说明 + 0.9303 -> 0.9304（EN/ZH 双稿）。"""
import io, sys

APPLY = '--apply' in sys.argv

JOBS = [
    ('docs/manuscript/latex/main.tex',
     '0.9536 and 0.9303, and are reported in Fig.~\\ref{fig:span}.',
     '0.9536 and 0.9304, and are reported in Fig.~\\ref{fig:span}. '
     'The overall ratio is an MSE-pooled quantity and is not the arithmetic mean '
     'of the per-energy ratios.'),
    ('docs/manuscript/latex/main_zh.tex',
     '0.9303，见图~\\ref{fig:span}。',
     '0.9304，见图~\\ref{fig:span}。汇总比值为误差加权池化值，不是逐能量比值的算术平均。'),
]

ok = True
for path, old, new in JOBS:
    s = io.open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    n = s.count(old)
    print('%-42s count=%d' % (path.split('/')[-1], n))
    if n != 1:
        ok = False
        continue
    if APPLY:
        s = s.replace(old, new)
        io.open(path, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
        print('   applied')

if not ok:
    print('FAIL: anchor not unique')
    sys.exit(1)
print('OK' if APPLY else 'check passed (use --apply)')
