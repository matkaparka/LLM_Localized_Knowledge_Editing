"""EC-11 arm names and the threshold vectors of the families that reuse KT's routed entries.

  none                 nothing written (no entries, no GRACE codebook, no retrieval)
  kt:n:m:f             KT: max(f, entry's largest similarity over the format context and the first n probe questions
                       + m); kt:2000:0.02:0.90 is RV-09's / KT-10's `matched`
  kthyg:n:m:f          the same, with entries whose subject field the hygiene rule flags switched off
  hyg:tau              hygiene-flagged entries off, every other entry tau
  rad:c                GRACE's radius rules on KT's entries (initial radius = the chord of cosine c), see rad_thresholds
  grace:l:sX / :aX     GRACE-IV at layer l, initial radius X times the probe scale (sX) or X absolute (aX)
  rag:..., ragp:m      retrieval (ec_rag)
"""
from __future__ import annotations

import torch

OFF = 2.0   # a threshold no centred cosine can reach


def family(arm: str) -> str:
    kind = arm.split(':', 1)[0]
    return {'none': 'none', 'kt': 'kt', 'kthyg': 'kt', 'hyg': 'hyg', 'rad': 'rad', 'grace': 'grace',
            'rag': 'rag', 'ragp': 'rag'}[kind]


def subfamily(arm: str) -> str:
    """For the secondary retrieval contrast: gated by one cut-off vs gated per record."""
    if arm.startswith('ragp:'):
        return 'ragp'
    if arm.startswith('rag:') and not arm.startswith('rag:always'):
        return 'rag_gated'
    return family(arm)


def parse_kt(arm: str) -> tuple[bool, int, float, float]:
    kind, n, m, f = arm.split(':')
    return kind == 'kthyg', int(n), float(m), float(f)


def parse_grace(arm: str) -> tuple[int, str, float]:
    _, layer, spec = arm.split(':')
    return int(layer), spec[0], float(spec[1:])


def kt_thresholds(pm: torch.Tensor, nq: int, n: int, margin: float, floor: float) -> torch.Tensor:
    """pm: probe_max table (nq probe rows, then the context row) over all entries. n >= nq means every probe (test
    streams keep slightly fewer than 2,000 probes after the leakage filter)."""
    n = min(int(n), nq)
    if n <= 0:
        raise ValueError(f'no probe questions ({nq} available)')
    return torch.clamp(torch.maximum(pm[:n].max(0).values, pm[nq]) + margin, min=floor)


def with_hygiene(thr: torch.Tensor, flagged: torch.Tensor) -> torch.Tensor:
    out = thr.clone()
    out[flagged] = OFF
    return out


def rad_thresholds(keys: torch.Tensor, labels: list, c: float) -> tuple[torch.Tensor, dict]:
    """GRACE's codebook rules on unit keys, in entry order, with chord distances d = sqrt(2 - 2 cos):
      nearest active entry j farther than r0 + r_j   -> the entry keeps r0
      nearer, different answer                       -> both shrink: r_j = d/2 - 1e-5, the new one d/2
      nearer, same answer                            -> r_j = max(r_j, d); the new entry is switched off
    Radii become cosine thresholds 1 - r^2/2; switched-off entries get OFF. Routing stays KT's rule (the most similar
    entry above its own threshold wins)."""
    r0 = (2.0 - 2.0 * c) ** 0.5
    n = keys.shape[0]
    radius = [None] * n
    active: list[int] = []
    splits = expansions = 0
    for i in range(n):
        if not active:
            radius[i] = r0
            active.append(i)
            continue
        cos = keys[torch.tensor(active, device=keys.device)] @ keys[i]
        d = torch.sqrt(torch.clamp(2.0 - 2.0 * cos, min=0.0))
        k = int(d.argmin())
        dmin, j = float(d[k]), active[k]
        if dmin > r0 + radius[j]:
            radius[i] = r0
            active.append(i)
        elif labels[i] != labels[j]:
            radius[j] = dmin / 2 - 1e-5
            radius[i] = dmin / 2
            active.append(i)
            splits += 1
        else:
            radius[j] = max(radius[j], dmin)
            expansions += 1
    thr = torch.full((n,), OFF, device=keys.device)
    for i in active:
        thr[i] = 1.0 - radius[i] ** 2 / 2.0
    return thr, dict(active=len(active), splits=splits, expansions=expansions)
