"""SP-16 post hoc (written 2026-11-23 after the test analysis; not part of the frozen protocol, no gate): the
trigger-frequency baseline against random inspection, with the frozen analysis's own bootstrap (same seed, same
resampling of streams and protected questions, so the draws are paired with those behind analysis.json).

  python checks/freq_vs_random_post_hoc.py RUN OUT_JSON

The frozen bootstrap reports intervals for risk - random and risk - frequency; here the frequency hits are passed in
the risk position, so its 'diff_random' is frequency - random and its 'rate_risk' is the frequency hit rate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sp_analysis as A  # noqa: E402


def main(run, out):
    run = Path(run)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    recs = A.load_run(run)
    res = {}
    for name, match in A.MATCHERS.items():
        arrays = [A.stream_block(rec, cfg, match)[1] for rec in recs.values()]
        res[name] = {}
        for k in cfg['budgets']:
            swapped = [dict(a, **{f'hit{k}': a[f'fhit{k}']}) for a in arrays]
            p = A.pooled(swapped, k)
            ci = A.bootstrap(swapped, k)
            res[name][str(k)] = dict(damaged=p['damaged'], freq_hits=p['hits'], random_expected=p['random_expected'],
                                     rate_freq=p['rate_risk'], rate_random=p['rate_random'],
                                     diff_freq_random=p['diff_random'],
                                     ci_rate_freq=ci['rate_risk'], ci_diff_freq_random=ci['diff_random'],
                                     draws_with_damage=ci['draws_with_damage'])
            print(name, k, json.dumps(res[name][str(k)]))
    doc = dict(run=run.name, note='post hoc, descriptive; not part of the frozen SP-16 protocol', draws=A.DRAWS,
               seed=A.SEED, results=res)
    Path(out).write_text(json.dumps(doc, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
