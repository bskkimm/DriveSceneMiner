# Hybrid stationary-obstacle avoidance

The experimental `--hybrid-avoidance` flag is available for `run`, `classify`
and offline `restitch`. It is disabled by default.

```bash
uv run jevsceneminer restitch out/original --out out/avoidance --hybrid-avoidance
```

The rule uses one sample's continuous reference lane, full object observations
and model predictions. A candidate needs a stopped tracked obstacle ahead with
positive footprint overlap with the reference lane, an opposite-side lateral
excursion of 0.5–4 m, a pass and a settled return near the same reference lane.
It requires complete model coverage of the candidate interval, at least one raw
winning avoidance prediction above `--min-prob`, and no conflicting raw maneuver.
Object observations must cover the maneuver with gaps no longer than one second.
The ego heading must remain within 15 degrees of the reference-lane tangent.

The rule requires available body geometry and known non-intersection map context.
Ambiguous overlapping map polygons, missing facts, incomplete windows, moving
objects and missing tracks cause abstention. A nearby object alone is insufficient.
Map overlap and clearance are geometric evidence and do not establish causation.

Accepted NOWs receive an avoidance correction after normal smoothing and before
optional indicator gating. Generic maneuver tail extension is disabled for
avoidance in this mode. Saved raw model answers, probabilities, facts and scripts
remain unchanged. Generator metadata records accepted samples, intervals,
obstacle track IDs and conflicts with other enabled hybrid rules. Scene support
remains the mean raw model support, not a hybrid probability.

Thresholds are preliminary. A cached Boston check found zero accepted samples
among 933 observations. Consequently, this rule has not demonstrated either
true-positive recovery or false-positive reduction on that recording. Evaluate
independent reviewed cases before relying on it. It does not reject a model
avoidance label solely because the rule abstains.
