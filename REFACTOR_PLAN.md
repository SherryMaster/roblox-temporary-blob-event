# Refactor plan

This repository already has a sound pure game model. The overhaul will keep the
model independent from vision, desktop input, and Qt, and will make the normal
product flow a single observed generation followed by deterministic local
execution.

## Preserve

- `Board` as an immutable canonical tuple of bottom-to-top, non-empty columns.
- Orthogonal component detection, logical `(row_from_bottom, column)` cells,
  gravity, left column compaction, deterministic `apply_move`, and `k²` scoring.
- Existing text-board CLI/developer solvers, Pillow vision fixtures, X11
  fallback, and the current safety-oriented interfaces where they remain useful.
- The current tests as regression coverage; new tests will validate complete
  plans rather than only individual solver objects.

## Replace or extend

- Replace the default beam-only product path with a portfolio `HybridPlanner`
  exposing Fast, Balanced, Deep, and Exhaustive quality presets. Greedy, beam,
  rollout, and exact implementations remain available for tests and benchmarks.
- Add immutable `Plan`, `PlanStep`, `RootMoveEvaluation`, and
  `PlanningSession`/generation state. A plan contains every simulated
  intermediate board and is replay-validated before it can be executed.
- Replace the old capture/calibrate/scan/re-read loop with `scan_and_solve`: one
  hidden capture supplies calibration and classification, then the resulting
  board is frozen. Normal manual and autoplay steps consume the frozen plan;
  only explicit Rescan/Recover or opt-in Verified execution captures again.
- Replace the Qt top-level overlay with a separate GTK3/PyGObject
  `gtk-layer-shell` service on Wayland. It will use the overlay layer, no
  keyboard interactivity, and an explicitly empty pointer input region. The
  control panel communicates with it through a small local JSON-lines socket.
  Qt remains the control-panel toolkit and X11 keeps its fallback overlay.
- Centralize Hyprland monitor discovery and conversion between capture/layout
  coordinates, logical layer-surface coordinates, and input coordinates.
- Replace the debug text-dump UI with a compact setup/plan/execution panel,
  custom painted `BoardWidget`, and plan timeline. Worker computation reports
  progress through Qt signals and never touches widgets directly.

## Why the current solver is too greedy

`BeamSolver` seeds a greedy incumbent and only replaces it when a beam node
reaches a terminal state. On a 10x10 board, a large beam can spend its complete
budget expanding several plies without completing a continuation, so the
greedy path remains the answer even though partial states may contain a better
future. The replacement search will attach a legal complete rollout to
promising partial nodes immediately, use that rollout as a real lower bound,
and compare complete terminal scores. Root branches will be explored fairly,
deduplicated, and ranked by full-game results rather than immediate group size.

Each search node will carry `score_so_far`, a complete continuation/lower
bound, and the admissible upper bound
`score_so_far + sum(remaining_color_count²)`. Exact late-game states will use a
memoized acyclic value function storing both future score and best move.

## Why the existing Qt overlay is unsuitable on Hyprland/Wayland

`QtOverlay` is a normal Qt top-level window with always-on-top/tool flags. Under
Wayland the compositor still treats it as a regular application surface; those
flags do not make it a shell layer and positioning is therefore compositor
managed. Transparency and `WA_TransparentForMouseEvents` also do not establish
the Wayland protocol's empty pointer input region. A native layer-shell surface
is required for a compositor-owned overlay that is above normal windows,
keyboard-inert, and genuinely click-through.

The first-class implementation will use Arch's `gtk-layer-shell` plus
`python-gobject` packages. It will target Hyprland/wlroots and document the
explicit X11/unsupported-compositor fallback. The overlay diagnostic command
will print monitor, scale, capture region, local layer region, and calculated
click coordinates and will be used for actual visual verification.

## Single-scan planning and frozen execution

`AutomationController.scan_and_solve()` hides the overlay, captures the configured
region exactly once, samples the same image, derives calibration if needed,
classifies all 100 cells, validates settled geometry, and freezes the initial
`Board`. Search runs only against that board. The resulting `Plan` includes
`before_board`, `move`, `after_board`, group size, immediate/cumulative score,
and screen click cell for every step. Construction asserts legal replay and
score conservation.

Normal `show_step`, `next_step`, and autoplay use only those frozen steps and
the configured animation delay. There is no screen capture, classification, or
solver restart in that path. `rescan_recover()` explicitly discards the old
remaining plan, captures the physical board, and builds a new complete plan.
Verified execution is a separately enabled diagnostic/safety mode and is off
by default.

## Delivery and verification

1. Add the plan/session and hybrid solver invariants while preserving core tests.
2. Add exact random-board comparisons, trap/column-collapse/gravity fixtures,
   real-generation benchmark data, and single-capture controller tests.
3. Add the coordinate module and native Wayland service/IPC with a manual
   `overlay-test` command; never silently substitute a normal Qt window on
   Wayland.
4. Redesign the UI and CLI, then update README/config/PLAN documentation.
5. Run the complete suite, deterministic benchmarks, real 10x10 benchmark,
   replay validation, and the live overlay diagnostic when a Hyprland session
   is available. If it is not available, report that limitation and leave the
   documented one-command local verification path.
