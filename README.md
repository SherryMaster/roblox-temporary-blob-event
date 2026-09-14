# Blob Solver

Blob Solver is a computer-vision-assisted SameGame planner for a colored-block
browser game. It reconstructs one generation, searches complete deterministic
continuations, and then executes the frozen plan without repeatedly reading the
screen.

The primary desktop target is Omarchy on Arch Linux with Hyprland and Wayland.
The game engine, planner, and vision code remain independent of the desktop
session and can be run entirely from text fixtures.

## Game model

- Orthogonal connectivity only; diagonals do not connect.
- A legal move removes a connected component of at least two cells.
- Each column falls downward after a removal.
- Empty columns are removed and the remaining columns shift left.
- A group of size `k` scores `k²`.

`Board` is immutable and canonical: non-empty columns are stored
bottom-to-top. Move cells use logical `(row_from_bottom, column)` coordinates.
The desktop layer converts those coordinates to screen positions only when a
highlight or click is needed.

## Install

Python 3.12 or newer is required.

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[test]'
```

For the optional control panel:

```sh
.venv/bin/pip install -e '.[ui]'
```

The `--system-site-packages` flag lets an Arch-created environment see the
pacman `python-gobject` and `python-cairo` modules. If your Python environment
is intentionally isolated, install equivalent GTK bindings inside that
environment instead.

Run the full automated suite with:

```sh
.venv/bin/pytest
```

### Arch / Omarchy packages

The first-class Wayland path uses the wlroots layer-shell protocol through
GTK3/PyGObject:

```sh
sudo pacman -S grim slurp ydotool gtk3 gtk-layer-shell python-gobject python-cairo
```

`ydotoold` must be running and able to access `/dev/uinput` for autoplay. On
systems where the daemon requires elevated uinput access, run it as a system
service or start `sudo ydotoold --touch-on`; confirm the installed daemon's
help output because Arch package versions differ. Scan and hint mode do not
need ydotool. Package names can vary with an Arch mirror; the required
capabilities are `grim`, `slurp`, `ydotoold`, GTK3, `python-gobject`,
`python-cairo`, and `gtk-layer-shell`.

The layer-shell client probes the control-panel interpreter and then the usual
Arch interpreters (`/usr/bin/python3`, `/usr/bin/python`) for GTK introspection,
so an isolated mise/venv Python can still launch the native service. To force a
specific interpreter, set `BLOB_SOLVER_WAYLAND_PYTHON` before starting the UI.

## Default single-scan workflow

The normal flow is deliberately generation-based:

1. Select a fixed board rectangle with `slurp` or configure it in TOML.
2. Press **Scan & Solve** in the UI, or run `blob-solver solve`.
3. The overlay is hidden and the region is captured exactly once.
4. The same image supplies calibration, all 100 classifications, and the
   immutable initial `Board`.
5. The hybrid planner builds a complete best-known sequence and simulates every
   intermediate board before publishing it.
6. **Show Step**, **Next Step**, and **Auto Play** consume that frozen sequence.

Normal execution performs no capture, classification, or solver restart after
each click. A configurable animation delay defaults to 350 ms.

The only normal ways to read the screen again are:

- **Rescan / Recover**, which discards the remaining plan and solves a new
  complete plan from a new physical observation.
- **Verified execution**, an advanced opt-in mode that restores capture-after-
  each-move validation for debugging or unusually unreliable conditions.

Manual mode uses **Next Step** to advance the predicted board after the user
clicks. **Previous Step** only inspects an earlier predicted state; it never
changes the physical game. An accidental click should use **Rescan / Recover**.

## Control panel

Start the compact PySide6 control panel with:

```sh
blob-solver ui
```

It has setup, plan, and execution areas. The plan area contains a painted
graphical board and a timeline. Selecting a future timeline row displays that
`PlanStep.before_board`, highlights the complete connected component, and shows
the step score and projected final score. Search progress reports actual
complete-plan scores, upper bounds, state counts, terminal plans, and elapsed
time; heuristic values are not presented as projected scores.

The normal quality choices are:

- **Fast** — roughly one second when the environment allows it.
- **Balanced** — roughly five seconds.
- **Deep** — roughly 20 seconds.
- **Exhaustive** — no normal short deadline; attempts a proof.

Exact timings depend on board shape and CPU. In the control panel, a zero
search-time override means “use the selected quality preset”; a positive value
provides an explicit bound. Root worker count is automatic by default (`0`),
serial when set to `1`, and otherwise bounded to the requested number.
Internal algorithm names are intended for developers and benchmarks, not
ordinary operation.

## Wayland overlay

On a Wayland session the control panel does not create the board overlay as a
Qt top-level window. The factory starts a separate GTK3 process using
`gtk-layer-shell`:

- the surface is on the compositor `OVERLAY` layer;
- it has no decorations, no exclusive zone, and no keyboard interactivity;
- the realized surface receives an explicitly empty Cairo/GDK input region;
- the entire connected group is outlined and softly filled;
- the actual click cell receives a stronger target ring;
- a small label shows the step, color, immediate score, and group size.

The empty pointer region is important: transparent pixels alone do not make a
surface click-through. The client waits for an IPC acknowledgement after each
show/hide command, so capture does not race a still-visible highlight. The
overlay communicates with the PySide6 process over a local Unix socket and
knows nothing about search algorithms.

The X11 Qt overlay remains only as the explicit X11 fallback. It is not used by
the Wayland factory path. When `WAYLAND_DISPLAY` is present, `backend = "auto"`
does not silently switch to X11 if a Wayland dependency is missing; choose
`backend = "x11"` explicitly if that is genuinely intended.

### Coordinates and scaling

Wayland capture pixels, slurp geometry, Hyprland layout coordinates, layer-shell
logical coordinates, and input coordinates are not assumed to be interchangeable.
`CoordinateMapper` is the single conversion module. It consumes
`hyprctl monitors -j`, records monitor position, logical dimensions, scale, and
focus, and logs:

- capture region;
- selected monitor and scale;
- local logical layer region;
- logical click point;
- calculated input click point and its declared coordinate space.

Select a region wholly inside one output, especially when using multiple
monitors or fractional scaling. `desktop.input_space = "logical"` is the
default. Set `physical` only when the selected input backend is known to expect
physical pixels.

Run the manual diagnostic on the real Hyprland session:

```sh
blob-solver overlay-test --backend wayland --region 100,200,820,740 --rows 10 --cols 10
```

It draws a numbered test grid, prints monitor/scaling diagnostics, and waits
while you verify alignment and click-through behavior. If the current execution
environment has no live Wayland compositor, this is the one command to run on
the Omarchy desktop.

## Text-board commands

Rows are written top-to-bottom. Compact strings and whitespace-separated tokens
are accepted; `.`, `_`, and `-` mean empty cells.

```sh
blob-solver inspect-board examples/real_generation_001.txt
blob-solver solve-board examples/real_generation_001.txt --solver hybrid --quality fast --time 1
blob-solver benchmark examples/real_generation_001.txt --time 2
blob-solver board-debug examples/real_generation_001.txt
```

The supplied fixture's two-second benchmark currently reports greedy `288`,
legacy beam `288`, and hybrid Fast/Balanced/Deep `798` on this machine;
rollout is intentionally seed/time sensitive. These are benchmark observations,
not hard-coded answers or optimality claims.

The developer solver choices are `greedy`, `greedy-size`, `beam`,
`legacy-beam`, `rollout`, `exact`, and `hybrid`. `legacy-beam` is the frozen
pre-overhaul baseline used only for comparison. Randomized rollout search is
called rollout; it is not mislabelled as MCTS.

The desktop commands are:

```sh
blob-solver select-region --backend wayland
blob-solver scan --backend wayland
blob-solver solve --backend wayland
blob-solver hint --backend wayland
blob-solver autoplay --backend wayland --confirm
blob-solver capture-fixture --backend wayland --output tests/fixtures/real-board-001.png
blob-solver overlay-test --backend wayland
blob-solver ui
```

`solve` captures once and prints the complete plan. `capture-fixture` stores
that one screenshot and its classified symbolic matrix for later regression
testing.

## Planner and plan invariants

The product-facing planner is `HybridPlanner`. It combines:

1. several immediate complete incumbents (score, topology, consolidation,
   small-removal, low-fragmentation, and seeded randomized policies);
2. fair exploration of every legal root move;
3. best-first/beam-style expansion with root diversity and state deduplication;
4. complete rollout continuations attached to promising partial nodes;
5. exact memoized suffix solving when block count, legal-group count, and work
   estimates make it practical;
6. an admissible color-count upper bound for safe pruning.

Independent root continuations may use the configured process pool. The pool is
used only when there are enough roots to amortize startup and serialization;
otherwise the planner stays serial. Worker inputs are immutable board/search
data, and an unavailable or failed multiprocessing start method falls back to
the same serial planner.

Every search node has a score-so-far, a complete lower-bound continuation, and
an upper bound. The optimistic bound is:

```text
score_so_far + sum(remaining_count[color]²)
```

An approximate result is reported as **BEST KNOWN PLAN**. **OPTIMAL PLAN
PROVEN** is used only after exhaustive branch completion or safe proof. The
reported total is always the sum of replayed step scores; the gap is the
difference between the best-known total and a safe upper bound.

`Plan` owns `PlanStep.before_board` and `PlanStep.after_board` for every move.
Construction verifies:

```text
every move is legal on its before_board
after_board == apply_move(before_board, move)
sum(step.immediate_score) == total_score
the final board has no legal group
```

The reference fixture contains the supplied 10x10 generation. Its immediate-
largest-group greedy baseline is independently checked at 288; the deterministic
Fast hybrid incumbent currently exceeds the supplied 436 screenshot result on
this fixture without encoding that score as a special case.

## Vision and real fixtures

Each cell uses a centered inner patch and per-channel median, reducing rounded
corner, outline, navy-background, and pointer contamination. Calibration uses
the same initial image as classification when no saved profile exists. Unknown
colors, low-confidence cells, and invalid settled-board holes reject the
generation before it can be planned or clicked.

Generated Pillow rectangles remain unit tests. To capture a real browser board:

```sh
blob-solver capture-fixture --backend wayland \
  --region 100,200,820,740 \
  --rows 10 --cols 10 \
  --output tests/fixtures/real-board-001.png
```

Commit the resulting PNG and symbolic matrix after visually checking them.
Real fixtures should cover rounded cells, gradients, the navy background,
browser zoom, selected outlines, and pointer contamination.

## Configuration and logging

Copy `config.example.toml` and adjust the fixed region, grid dimensions,
confidence threshold, quality preset, search limits, animation delay, and
`verified_execution`. The last successful region and calibration are persisted
when a config path is provided.

Session events are JSON Lines and include captures, board hashes, plan/search
statistics, recommendations, coordinate diagnostics, and explicit verification
results. Auto Play is an explicit UI/CLI action rather than a startup config
flag. There is no decorative `max_mismatches` setting: mismatch handling is
implemented only by the opt-in Verified path.

## References

The Wayland implementation follows the compositor protocol and upstream
projects:

- [wlr-layer-shell protocol](https://github.com/swaywm/wlroots/blob/master/protocol/wlr-layer-shell-unstable-v1.xml)
- [gtk-layer-shell](https://github.com/wmww/gtk-layer-shell)
- [Hyprland monitor configuration](https://wiki.hypr.land/Configuring/Basics/Monitors/)
- [grim](https://github.com/emersion/grim)
- [slurp](https://github.com/emersion/slurp)
- [ydotool](https://github.com/ReimuNotMoe/ydotool)
