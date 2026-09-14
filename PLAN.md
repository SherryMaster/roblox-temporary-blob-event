# Blob Solver implementation plan

This project is being built in milestones so the game model can be trusted before
it is allowed to drive a real browser.

## Internal representation

`Board` is a frozen dataclass containing:

```text
tuple[tuple[Color, ...], ...]
```

Each inner tuple is one non-empty column, ordered **bottom-to-top**. Empty columns
are omitted. Consequently a board is already gravity-normalized and column-
compacted, making it cheap to hash and compare. `Color` is any hashable symbolic
identifier; the vision layer uses strings such as `red`, while text fixtures may
use `R`, `Y`, `G`, and `B`.

`Move.cells` contains `(row_from_bottom, column)` pairs. This is a logical game
coordinate, not a screen coordinate. The application maps it to a configured
fixed-size grid only at the moment it draws a hint or clicks.

Matrix conversion uses the conventional top-to-bottom visual order and accepts
`None` or `.` as empty cells. Strict conversion rejects internal holes, because
such a matrix cannot be a valid settled SameGame state.

## Milestones

- [x] Milestone 1 — immutable board, groups, scoring, gravity, compaction, and tests
- [x] Milestone 2 — exact memoized solver and optimality tests
- [x] Milestone 3 — greedy, beam, branch-and-bound, transposition table, rollout, heuristics, benchmark command
- [x] Milestone 4 — region/grid sampling, dynamic calibration, empty detection, confidence, settle detection, vision tests
- [x] Milestone 5 — Wayland/X11 capture and input interfaces, hint controller, overlay with preview fallback
- [x] Milestone 6 — safe autoplay, settle/readback, predicted-vs-actual verification, stop handling
- [x] Milestone 7 — PySide6 control panel, CLI, logging, fixtures, and documentation

## Safety invariants

1. The desktop layer never enters the solver.
2. An uncertain or unknown visual cell produces no playable board and cannot be
   clicked automatically.
3. Autoplay exposes one move at a time, waits for a settled observation, and
   compares it with the simulated next state.
4. A physical mismatch stops autoplay after a fresh observation and is logged.
5. Approximate solvers report “best known”; only a fully completed exact search
   reports “optimal proven”.

## Verification order

The pure engine and exact solver are tested first. Vision tests use generated
Pillow images and small perturbations. Desktop tests use fake capture/input
objects, so they do not require a Wayland or X11 session.
