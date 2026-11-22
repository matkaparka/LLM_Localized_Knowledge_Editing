"""XR-13 post-hoc description, written after the test results were seen; not part of the frozen analysis and not a
gate. In npm the keys that broke most facts were anchored by the extractor on the registry's name ('npm'), not on the
package. Per family this reports:

  - every entry whose extractor anchor token is not the gold anchor token (gold = the last occurrence of the entity
    name in the write question, as in xr_run.gold_span), as 'outside the name' (the anchor token is not one of the
    name's tokens) or 'inside the name' (a token of the name, but not its last one);
  - for those entries: the matched threshold and the facts each breaks on its own under the three intervention arms;
  - under each intervention arm at T: damaged facts, those with a culprit among the outside-the-name entries (a culprit
    breaks the fact on its own), those whose culprits are all such entries, and those that come back right when one
    such entry is switched off.

Also: leave-one-family-out means of the gates' quality differences, matched against every uniform point on the test
records (the frontier, with and without npm), and the unchanged dev rule (xr_analysis.tune) on dev subsets.

  python xr_posthoc.py TEST_RUN STREAMS_DIR OUT.json [DEV_RUN]
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'field_prefix_ablation_012'))
sys.path.insert(0, str(HERE))       # python -I does not put the script's folder on the path
import fp_data  # noqa: E402

ARMS = ('xr:uniform:0.90', 'xr:shuffled:0', 'xr:matched')


def load(run: Path) -> dict:
    out = {}
    for p in sorted(run.glob('stream_*.json.gz')):
        with gzip.open(p, 'rb') as g:
            out[p.name[7:-8]] = json.loads(g.read().decode('utf-8'))
    return out


def anchors(tok, question: str, span, entity: str):
    q = question.strip()
    text = fp_data.bare_prompt(question)
    off = len(fp_data.Q_PREFIX)
    k = q.rfind(entity)
    if not span or k < 0:
        return None
    ext = fp_data.token_at(tok, text, off + span[1] - 1)
    first, last = fp_data.token_at(tok, text, off + k), fp_data.token_at(tok, text, off + k + len(entity) - 1)
    kind = 'same' if ext == last else ('inside the name' if first <= ext < last else 'outside the name')
    ids = tok.encode(text, add_special_tokens=False)
    return dict(kind=kind, anchor_token=tok.decode([ids[ext]]), gold_token=tok.decode([ids[last]]))


def main(test_run: str, streams_dir: str, out_path: str, dev_run: str | None = None):
    from transformers import AutoTokenizer
    recs = load(Path(test_run))
    cfg = json.loads((Path(test_run) / 'config.json').read_text(encoding='utf-8'))
    tok = AutoTokenizer.from_pretrained(str(HERE.parent.parent / 'models' / 'qwen3_0.6b_base'), local_files_only=True)
    res = {}
    for name, rec in recs.items():
        d = json.loads((Path(streams_dir) / f'{name}.json').read_text(encoding='utf-8'))
        by_uid = {e['uid']: e for e in d['edits']}
        uids = rec['tables']['uids']
        thr = rec['thresholds']['matched']
        answers = rec['loc_answers']
        bad = {}
        for j, u in enumerate(uids):
            e = by_uid[u]
            a = anchors(tok, e['update'], e.get('span'), e['entity'])
            if a and a['kind'] != 'same':
                bad[j] = dict(entry=j, uid=u, entity=e['entity'], span_text=e.get('span_text'), **a,
                              matched_threshold=thr[j])
        outside = {j for j, b in bad.items() if b['kind'] == 'outside the name'}
        arms = {}
        for arm in ARMS:
            ivs = rec['interventions'].get(arm, [])
            n_with = n_only = n_restored = 0
            for r in ivs:
                if 'alone' not in r:
                    continue
                right = answers[str(r['i'])]
                culprits = {e for e, ans in r['alone'] if ans != right}
                if culprits & outside:
                    n_with += 1
                if culprits and culprits <= outside:
                    n_only += 1
                if any(e in outside and ans == right for e, ans in r.get('single', [])):
                    n_restored += 1
                for e in culprits & set(bad):
                    bad[e].setdefault('breaks_alone', {}).setdefault(arm, 0)
                    bad[e]['breaks_alone'][arm] += 1
            arms[arm] = dict(damaged=len(ivs), with_outside_culprit=n_with, only_outside_culprits=n_only,
                             restored_by_switching_one_off=n_restored)
        res[name] = dict(entries=len(uids), known=rec['screening']['known'],
                         inside_the_name=sum(1 for b in bad.values() if b['kind'] == 'inside the name'),
                         outside_the_name=len(outside), bad_entries=list(bad.values()), arms=arms)
    # leave-one-family-out means of the gates' other quantities (the analysis reports this only for Delta)
    import fp_analysis as fa
    sel_arm = (cfg.get("selected_uniform") or {}).get("arm", "uniform:0.90")
    sel = f"xr:{sel_arm}"
    measured = [n for n in recs if recs[n]['screening']['known'] >= 30]
    per = dict(delta={n: float(fa.delta_items(recs[n], 'xr').mean()) for n in measured},
               delta_u={n: float(fa.delta_items(recs[n], 'xr', sel).mean()) for n in measured},
               q={n: float(fa.quality_items(recs[n], 'xr', 'cur').mean()) for n in recs},
               r={n: float(fa.quality_items(recs[n], 'xr', 'ret').mean()) for n in recs},
               q_u={n: float(fa.quality_items(recs[n], 'xr', 'cur', sel).mean()) for n in recs},
               r_u={n: float(fa.quality_items(recs[n], 'xr', 'ret', sel).mean()) for n in recs})
    loo = {f: {k: fa.mean([v for n, v in per[k].items() if n != f]) for k in per} for f in recs}
    # matched against every uniform point on these records (the test frontier), on all families and without the
    # families whose damage under 0.90 is mostly from keys anchored outside the package name (more than half of the
    # damaged facts with such a culprit), then also without every family that has such keys at all
    dominated = sorted(n for n, v in res.items() if v['arms']['xr:uniform:0.90']['damaged']
                       and v['arms']['xr:uniform:0.90']['with_outside_culprit'] > 0.5 * v['arms']['xr:uniform:0.90']['damaged'])
    any_outside = sorted(n for n, v in res.items() if v['outside_the_name'])
    subsets = {'all': [], 'without ' + '+'.join(dominated): dominated, 'without ' + '+'.join(any_outside): any_outside}
    reps, seed = int(cfg['rule']['bootstrap']), int(cfg['rule']['bootstrap_seed'])
    front, sub_summary = {}, {}
    for label, drop in subsets.items():
        fam = [n for n in recs if n not in drop]
        mfam = [n for n in measured if n not in drop]
        rows = {}
        for g in sorted(a for a in next(iter(recs.values()))['arms'] if a.startswith('xr:uniform:')):
            rows[g] = dict(delta_u=fa.mean([float(fa.delta_items(recs[n], 'xr', g).mean()) for n in mfam]),
                           q_diff=fa.mean([float(fa.quality_items(recs[n], 'xr', 'cur', g).mean()) for n in fam]),
                           r_diff=fa.mean([float(fa.quality_items(recs[n], 'xr', 'ret', g).mean()) for n in fam]),
                           damage=fa.mean([fa.arm_metrics(recs[n], g)['damage'] for n in mfam]))
        rows['xr:matched'] = dict(damage=fa.mean([fa.arm_metrics(recs[n], 'xr:matched')['damage'] for n in mfam]),
                                  Q=fa.mean([fa.arm_metrics(recs[n], 'xr:matched')['Q'] for n in fam]),
                                  R=fa.mean([fa.arm_metrics(recs[n], 'xr:matched')['R'] for n in fam]))
        front[label] = rows
        sub_summary[label] = dict(
            families=fam, measured=mfam,
            delta=fa.summary({n: fa.delta_items(recs[n], 'xr') for n in mfam}, reps, seed),
            delta_u=fa.summary({n: fa.delta_items(recs[n], 'xr', sel) for n in mfam}, reps, seed),
            q=fa.summary({n: fa.quality_items(recs[n], 'xr', 'cur') for n in fam}, reps, seed),
            r=fa.summary({n: fa.quality_items(recs[n], 'xr', 'ret') for n in fam}, reps, seed),
            q_u=fa.summary({n: fa.quality_items(recs[n], 'xr', 'cur', sel) for n in fam}, reps, seed),
            r_u=fa.summary({n: fa.quality_items(recs[n], 'xr', 'ret', sel) for n in fam}, reps, seed),
            # the uniform points the unchanged dev rule picks once npm_dev (and cran_dev) are left out, and 0.96
            other_uniform={g: dict(
                delta_u=fa.summary({n: fa.delta_items(recs[n], 'xr', g) for n in mfam}, reps, seed),
                q_u=fa.summary({n: fa.quality_items(recs[n], 'xr', 'cur', g) for n in fam}, reps, seed),
                r_u=fa.summary({n: fa.quality_items(recs[n], 'xr', 'ret', g) for n in fam}, reps, seed))
                for g in ('xr:uniform:0.92', 'xr:uniform:0.94', 'xr:uniform:0.96')})
    # the dev rule (xr_analysis.tune, unchanged) on dev subsets, to see what shaped the fallback choice
    dev_subsets = None
    if dev_run:
        import xr_analysis
        dev = {n: r for n, r in load(Path(dev_run)).items()}
        dcfg = json.loads((Path(dev_run) / 'config.json').read_text(encoding='utf-8'))
        dev_subsets = {}
        for drop in (None, 'npm_dev', 'cran_dev', ('npm_dev', 'cran_dev')):
            dd = tuple(drop) if isinstance(drop, tuple) else (drop,)
            t = xr_analysis.tune({n: r for n, r in dev.items() if n not in dd}, dcfg)
            dev_subsets['all' if drop is None else 'without ' + '+'.join(dd)] = dict(arm=t['arm'], flag=t['flag'])
    out = dict(note='post hoc, after the test results; descriptive only', run=Path(test_run).name,
               selected_uniform=sel_arm, families=res, per_family_means=per,
               leave_one_family_out=loo, subsets=sub_summary, test_frontier=front, dev_rule_on_subsets=dev_subsets)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for n, v in res.items():
        print(n, 'outside', v['outside_the_name'], 'inside', v['inside_the_name'],
              {a: (x['damaged'], x['with_outside_culprit'], x['only_outside_culprits']) for a, x in v['arms'].items()})
        for b in v['bad_entries']:
            if b['kind'] == 'outside the name':
                print('   ', b['entity'], repr(b['span_text']), repr(b['anchor_token']), f"{b['matched_threshold']:.3f}",
                      b.get('breaks_alone'))


if __name__ == '__main__':
    main(*sys.argv[1:5])
