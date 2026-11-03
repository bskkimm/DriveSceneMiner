# Hybrid left/right lane changes (experimental)

Opt in with `--hybrid-lanechanges` on `run`, `classify` or offline `restitch`:

```bash
uv run jevsceneminer restitch out/original --out out/hybrid-lanes --hybrid-lanechanges
```

Use a separate output folder to retain the original scenes. The flag makes no
additional API calls. It can be combined with `--hybrid-turns`; disagreement at a
timestamp causes both corrections to abstain, with a conflict count in the audits.

## Evidence and decision

The rule evaluates each NOW using its complete `facts.ego_samples` window. It
never joins offsets across windows: each window has its own continuous reference,
anchored to its first observed lane chain. It does not use the reset-prone offset
from the currently matched segment to establish lateral movement.

- Uniquely connected non-intersection successors with a reciprocal unique
  predecessor form a map-derived lane chain. Fork alternatives are not unioned.
- Origin and target must be reciprocal same-direction left/right neighbors at
  the observed transition. A successor transition alone is not a lane change.
- The reference chain must match the origin. Require two stable origin
  observations before movement and three settled target observations afterward.
- Require 1.5–5 m signed lateral displacement, at least two movement increments,
  lateral rates no greater than 2 m/s, forward speed at least 0.5 m/s during
  movement, and heading range at most 20 degrees. Small reverse jitter up to
  0.1 m per step is tolerated. Stable-offset tolerance is 0.15 m.
- Require boundary-crossing evidence on the correct side. Origin and target
  footprint shares must progress from at least 80%/at most 10% to the reverse.
  Shares are conservative per-chain maxima, never sums of overlapping polygons
  or claims of exclusive physical ownership.
- Require valid body dimensions, join gaps at most 0.5 m, native ambiguous
  overlap at most 5%, unmapped footprint at most 10%, and non-intersection
  matched road observations throughout the window. Uniform table cadence must
  be at most one second; missing observations or malformed evidence abstain.
- At least one raw winning model lane-change label in the movement interval
  must have finite support at least `--min-prob`. Other raw maneuvers inside that
  interval block corrections. All-frame model misses are not recovered.

The accepted interval covers first moved to last moved observations, with the
existing half-inference-step cells. Apply corrections after smoothing/confidence
filtering and before optional indicator gating. Generic lane-change tails are
disabled in this mode to avoid extending beyond settling evidence. Speed phases
retain the existing merge behavior.

Stable same-chain stationary triplets (speed at most 0.1 m/s, offset span at most
0.1 m, footprint share at least 80%) veto lane-change labels, including smoothing
holes. Ambiguous stationary map flips cause abstention rather than invented lane
ownership. Manual lane reviews that remove native geometry remain unknown here.
Stationary vetoes apply only to merged lane-change labels; they cannot erase
other maneuver labels introduced by smoothing. Positive lane-change corrections
also cannot overwrite another merged maneuver.

## Compatibility and audit

Preprocessing adds `left_neighbor_id`, `right_neighbor_id`, `previous_ids` and
successor `id` fields to `facts.lanes`; the readable Jev script stays unchanged.
Older inputs without neighbor topology cannot establish a positive correction.
The original and target chain roots in the audit are map IDs local to the complete
candidate window, not global physical-lane annotations.

`generator.hybrid_lanechanges` records version, thresholds, accepted/veto counts,
intervals, source/target chains and reasons. Saved raw labels, probabilities,
facts and scripts remain unchanged during merging. Scene support remains a raw
mean, not a new hybrid probability, and may be below the confidence threshold.

This is synthetic-test-validated experimental logic, not a measured accuracy
improvement. Correlated map/log errors can still fool multiple geometric checks.
Fast/slow changes, partial moves, intersections, shoulders, incomplete windows,
multiple changes away from the window's reference chain and insufficient body
coverage can abstain. Reprocess original inputs for the new topology fields;
independent GT evaluation is required before changing defaults.
