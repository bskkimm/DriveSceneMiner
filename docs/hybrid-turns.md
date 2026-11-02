# Hybrid left and right turns (experimental)

`--hybrid-turns` combines saved model labels with canonical numerical NOW facts.
It is off by default and makes no extra API requests. Try it without inference:

```bash
uv run jevsceneminer restitch out/original --out out/hybrid-turns --hybrid-turns
uv run jevsceneminer view out/original --run jev=scenes --run hybrid=/absolute/path/to/out/hybrid-turns/scenes
```

Use a separate output directory to retain the original scenes. `run` and
`classify` also accept the flag. Old inputs without `facts.ego_samples` remain
unchanged. The readable script is never parsed to infer missing measurements.

## Decision rule

1. Find consecutive NOW observations rotating in the same direction at at least
   3 degrees/s and moving at least 0.5 m/s.
2. Require valid, low-rotation/stopped observations on both sides, no missing
   observations or cadence gaps, and integrated rotation from 40 to under 135
   degrees. Rotation is integrated at the inference cadence, not the table rate.
3. Require at least one raw winning model label for that direction with support
   at least `--min-prob` (default 0.4). All other raw labels in the bout must be
   that turn or `keep_lane`; other maneuvers block correction.
4. Require known map occupancy at every rotating NOW and at least one intersection
   observation. Every intersecting observation must have the matching mapped turn
   direction. If so, assign the turn throughout that rotation bout.
5. If the same complete rotation bout is entirely on matched non-intersection
   road, assign `keep_lane` across the whole bout, including any turn labels that
   smoothing introduced at originally keep-lane observations.

Missing map matches, unknown/contradictory connector directions, truncated bouts,
U-turn-sized rotation and unsupported model decisions cause abstention. A missing
intersection is not assumed to be a road. No lane IDs, polygon shares or indicator
state alone establish a turn. Existing optional indicator gating still applies.
Malformed fact containers and non-boolean map occupancy remain unknown. Invalid
model probabilities cannot provide turn support.

```text
NOW:         0  1  2  3  4  5  6  7
rotation:    .  .  R  R  R  R  .  .
raw model:   K  K  K  R  K  K  K  K
mapped turn:       R  R  R  R
hybrid:      K  K  R  R  R  R  K  K
K = keep_lane; R = turn_right
```

Corrections apply after existing smoothing/confidence filtering. They bypass
those two filters once jointly supported, then retain optional indicator gating.
Boundaries are the half-step inference cells around the accepted observations,
clipped to the recording. Generic `--maneuver-tail` extension is disabled for
left/right turns when hybrid mode is enabled. Other maneuvers retain their tails.
Motion and map occupancy boundaries are different; map entry/exit is not used as
the maneuver boundary.

## Evidence and limitations

Raw labels, raw probabilities, scripts and facts remain unchanged. Scene mean
probability continues to summarize raw model support for the final label; it is
not a new hybrid probability and can be below `--min-prob`. Longitudinal phases
continue to come from the original longitudinal predictions.

`generator.hybrid_turns` records the rule version, numerical thresholds, accepted
turn count, road-curve rejection count and decision intervals/reasons. Audit
intervals identify first/last NOW centers; scene boundaries use half-step cells.

This first version has synthetic regression coverage, not measured driving-data
accuracy. Map errors can still produce errors. Low-rate integration, slow turns,
pauses mid-turn, roundabouts and gradual/compound turns can cause abstention.
The rule does not find a turn if every raw model observation missed it. It does
not trim model turn labels outside a supported bout. Evaluate complete independent
GT intervals for label and boundary errors before changing defaults or claiming
accuracy gains. Lane changes, avoidance, branches and U-turns are outside scope.
