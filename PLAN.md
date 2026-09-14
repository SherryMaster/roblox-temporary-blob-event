# Blob Solver implementation plan

This repository is an existing implementation, not a greenfield prototype.
The immutable engine and its transition tests are the foundation. Desktop
integration is deliberately kept outside that engine.

## Canonical game state

`Board` stores a tuple of non-empty columns, each bottom-to-top. This makes
gravity a filter operation, left compaction deterministic, and equivalent
settled states hashable. `Move.cells` uses logical `(row_from_bottom, column)`
coordinates. Matrix conversion is top-to-bottom and strict conversion rejects
holes.

Preserved and regression-tested contracts:

- immutable canonical `Board`;
- orthogonal connected components;
- legal groups of size at least two;
- downward gravity;
- empty-column removal and left compaction;
- deterministic `apply_move`;
- square scoring.

## Current architecture

### One observed generation

`AutomationController.scan_and_solve()` hides the overlay, captures the selected
region once, samples that same image, derives calibration if necessary,
classifies all cells, validates the reconstruction, and freezes a
`GameGeneration`. Search then operates only on that immutable board.

`PlanningSession` owns the fixed region, captured generation, solved generation,
the execution cursor, and a separate predicted-state view cursor. `Plan` owns a replay-validated complete sequence;
`PlanStep` owns both intermediate boards, the move, actual score, cumulative
score, and logical click cell.

Normal `show_step()`, `next_step()`, `previous_step()`, and autoplay do not
capture, classify, or call `solve()` again. `rescan_recover()` is the explicit
replacement-generation path. Verified execution is separate and disabled by
default.

### Planner

`HybridPlanner` is the default product solver. Its portfolio is:

1. complete diverse incumbents;
2. fair full-continuation evaluation of every root move;
3. best-first/beam-style tree exploration with root diversity and deduplication;
4. complete continuation rollouts attached to partial states;
5. exact memoized late-game suffixes;
6. admissible color-count branch-and-bound pruning.

Each node has a real complete lower bound and a safe upper bound. The planner is
anytime: even a zero/tiny tree budget still returns a complete terminal plan
from Phase A, and cancellation returns the best complete plan found so far.

`ExactSolver` stores `exact_future_score` and `best_move` in its
`TranspositionTable`. Since every legal move removes at least two cells, the
state graph is acyclic by block count and those entries are safe to reuse.

Independent root continuations use a bounded process pool when the root count
is large enough to justify startup/serialization overhead; small boards and
platforms without a usable start method use the serial path.

## Desktop architecture

The primary path is Omarchy/Hyprland/Wayland:

```text
PySide6 control panel
        |
        +-- local Unix socket --> GTK3 gtk-layer-shell overlay service
        |
        +-- grim capture / slurp selection / ydotool input
```

The layer-shell service is a separate process, uses the compositor overlay
layer, is keyboard-inert, and sets an explicitly empty pointer input region.
The Qt overlay remains only for the X11 fallback. Coordinate conversions are
centralized in `desktop/coordinates.py` and use `hyprctl monitors -j` data.

## UI and state machine

The control panel is organized as Setup, Plan, Execution, and collapsible
Advanced settings. `BoardWidget` paints the reconstructed and predicted board;
the timeline selects future `PlanStep.before_board` states. Search progress
reports complete best-known scores, safe upper bounds, state counts, terminal
plans, and elapsed time.

Application states are explicit:

```text
NO_REGION -> READY_TO_SCAN -> SCANNING -> SEARCHING -> PLAN_READY
                                      -> SHOWING_STEP -> AUTOPLAY / PAUSED
                                      -> RECOVERY -> READY_TO_SCAN
```

Invalid operations such as autoplay without a plan, overlapping scans, and
advancing past plan completion are rejected.

## Quality presets

The normal product choices are `fast`, `balanced`, `deep`, and `exhaustive`.
Greedy, beam, rollout, and exact modes remain available for developer
comparison and benchmarks. Internal mode names are not required of normal
users.

## Fixtures and verification checklist

- `examples/real_generation_001.txt` is the supplied 10x10 reference board.
- Its immediate-largest-group greedy score is a regression baseline of 288.
- The benchmark command compares greedy, the preserved legacy beam, rollout, all hybrid quality
  presets, and exact when feasible.
- Small random boards compare `ExactSolver` to brute force.
- Every public plan is replay-validated.
- Controller tests assert one default capture and explicit second capture only
  for recovery.
- `capture-fixture` records real screenshots and expected matrices.
- `overlay-test` is the manual Hyprland alignment/click-through diagnostic.

Run the live diagnostic on the target desktop when a compositor is available:

```sh
blob-solver overlay-test --backend wayland --region 100,200,820,740 --rows 10 --cols 10
```

If the development environment has no live Wayland display, that command is the
remaining local verification step; the implementation must not substitute a
normal Qt window and call it a layer overlay.
