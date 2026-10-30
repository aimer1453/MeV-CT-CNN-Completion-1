# -*- coding: utf-8 -*-
"""仅在 tab:spanenergy 段落内替换 Image 四行（三种子），并改表注。"""
import io, re
ROWS = {2: [0.9106, 1.0088, 1.0093, 1.0104],
        4: [0.9396, 0.9481, 0.9486, 0.9533],
        6: [0.9306, 0.9389, 0.9412, 0.9438],
        8: [0.9311, 0.9340, 0.9395, 0.9420]}
EOL = chr(92) * 2

CFG = (('docs/manuscript/latex/main.tex', 'Image',
        'Row-16 entries are three-seed means (seed SD 0.003--0.053); the image-domain '
        'entries are single-run point estimates.',
        'All entries are three-seed means (Row-16 seed SD 0.003--0.053).'),
       ('docs/manuscript/latex/main_zh.tex', '\u56fe\u50cf',
        '\u7b2c 16 \u884c\u6761\u76ee\u4e3a\u4e09\u79cd\u5b50\u5747\u503c\uff08\u79cd\u5b50 SD 0.003--0.053\uff09\uff1b'
        '\u56fe\u50cf\u57df\u6761\u76ee\u4e3a\u5355\u6b21\u8fd0\u884c\u7684\u70b9\u4f30\u8ba1\u3002',
        '\u5404\u6761\u76ee\u5747\u4e3a\u4e09\u79cd\u5b50\u5747\u503c\uff08\u7b2c 16 \u884c\u7684\u79cd\u5b50 SD \u4e3a 0.003--0.053\uff09\u3002'))

for pfx, dom, old_note, new_note in CFG:
    s = io.open(pfx, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    a = s.index('\\label{tab:spanenergy}')
    b = s.index('\\label{tab:cubic}')
    head, body, tail = s[:a], s[a:b], s[b:]
    n = 0
    for sd, v in ROWS.items():
        pat = re.compile(r'^    (%s) & %d & [^\n]*$' % (dom, sd), re.M)
        ms = pat.findall(body)
        assert len(ms) == 1, (pfx, sd, len(ms))
        repl = '    %s & %d & %s %s' % (dom, sd, ' & '.join('%.4f' % x for x in v), EOL)
        body = pat.sub(lambda m, _r=repl: _r, body, count=1)
        n += 1
    assert body.count(old_note) == 1, (pfx, body.count(old_note))
    body = body.replace(old_note, new_note)
    io.open(pfx, 'w', encoding='utf-8', newline='').write((head + body + tail).replace('\n', '\r\n'))
    print(pfx, 'rows', n, 'ok')
