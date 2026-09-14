# Blob Solver

Blob Solver is a modular SameGame assistant for a colored-block browser game. It
captures a user-selected board rectangle, reconstructs the regular grid with
computer vision, searches for a high-scoring sequence, highlights only the next
logical move, and can optionally perform one verified click at a time.

The core game engine and all solver modes are independent of a desktop session.
Wayland/Hyprland integration is isolated behind adapters, so the same board model
can be tested from text fixtures or used with another capture/input backend later.

## Game model

- Orthogonal connectivity only; diagonals do not connect.
- A legal move removes a connected component of at least two cells.
- Every column falls downward independently.
- Empty columns are removed and surviving columns shift left.
- A group of size k scores k squared.

The frozen solver state is a tuple of non-empty columns, each stored
bottom-to-top. Move coordinates are (row_from_bottom, column) logical
coordinates, never desktop pixels.

Greedy largest-group play is only a baseline. The beam, rollout, and exact
branch-and-bound solvers search alternatives because a small move can cause
gravity and column collapse to merge later groups. A result is labeled
BEST KNOWN SOLUTION unless a completed exact search proves the optimum.

## Install

Python 3.12 or newer is required.

    python -m venv .venv
    .venv/bin/pip install -e '.[test,desktop]'

For the graphical control panel and transparent overlay:

    .venv/bin/pip install -e '.[ui]'

Run tests with:

    .venv/bin/pytest

## Text-only development tools

The text path is the safest way to understand the physics before using a real
browser:

    blob-solver inspect-board examples/board_small.txt
    blob-solver inspect-board examples/board_small.txt --move 0
    blob-solver board-debug
    blob-solver solve-board examples/board_demo.txt --solver exact --time 10
    blob-solver benchmark examples/board_demo.txt --time 2

Rows are written top-to-bottom. Use one-character colors such as R, Y, G, and B;
whitespace-separated symbolic tokens also work. A dot is empty. The validator
prints every legal group, its cells, and its square score.
board-debug also accepts pasted rows on stdin and lets you apply move indexes
interactively, printing the resulting board after every move. benchmark reports
the gap to the exact score when the exact run proves one.

## Wayland/Hyprland workflow

The preferred backend uses:

- grim to capture the exact selected rectangle
- slurp to interactively select a rectangle
- ydotool plus its ydotoold uinput daemon for optional absolute pointer input

On Arch, install the relevant packages using your normal package manager, then
ensure ydotoold has access to /dev/uinput. A capture-only or hint-only session
does not need ydotool. The application checks commands at runtime and reports
missing dependencies; it does not silently fall back to uncontrolled clicks.

Select and save a board region:

    blob-solver select-region

Or configure one directly:

    blob-solver --config config.toml hint --region 100,200,820,740 --rows 10 --cols 10

The last successful region is saved to
~/.config/blob-solver/config.toml unless --config is supplied.

If a transparent global overlay is unavailable under a compositor/session,
PySide6 hint mode still presents the reconstructed board and recommendation in
the control panel. The overlay is hidden around every capture so its outline can
never become a sampled color.

For X11, request --backend x11. Capture uses optional mss first and Pillow
ImageGrab as a fallback; input uses xdotool. Explicit coordinates are
recommended when no X11 region selector is installed.

## UI workflow

Start the control panel:

    blob-solver ui

1. Select or enter the board rectangle and set rows/columns.
2. Press Calibrate, then inspect the reconstructed matrix and confidence.
3. Press Analyze or Show Next Move.
4. Manually click the highlighted group, then press Re-read After Manual Click.
5. Enable Auto Play only after the observed transitions match the simulator.

Autoplay defaults off. It refuses to click when any cell is unknown or below the
confidence threshold, hides the overlay before input, waits for repeated stable
observations after each click, and compares the actual board with the simulated
next state. A mismatch causes a fresh replan and stops further automatic input.
The Stop and Emergency Stop controls are separate; Escape can be wired to the
same emergency method in the included control panel.

The CLI equivalents are:

    blob-solver scan
    blob-solver solve
    blob-solver vision-debug --output board-debug.png
    blob-solver hint
    blob-solver autoplay --confirm

The confirm flag is required for CLI autoplay as an intentional safety gate.

## Vision and calibration

The board rectangle is divided into equal cells. Each cell uses a centered inner
patch and a per-channel median, avoiding rounded corners, outlines, grid
background, and most pointer contamination. Calibration clusters bright samples
and matches clusters to the default red/yellow/green/blue symbolic labels. Dark
navy samples are classified as empty. Classification returns every cell's RGB,
symbol, and confidence.

An observation is rejected if it contains unknown colors, low-confidence cells,
or internal holes that violate settled-board geometry. vision-debug prints a
labeled matrix and can save a graphical preview with confidence values.

After a click, settle detection captures around a configurable interval and
requires the same valid canonical board for a configurable number of consecutive
frames. It does not rely on one fixed sleep.

## Solver modes

- greedy / greedy-size: baselines for comparison.
- beam: configurable anytime beam search with a transposition table and
  topology-aware one-ply ordering.
- rollout: deterministic-seed randomized greedy restarts.
- exact: depth-first branch-and-bound with memoized reaching states and the
  admissible upper bound sum(color_count squared).

All non-exact modes retain a complete greedy incumbent when a short wall-clock
budget expires. Exact mode reports OPTIMAL SOLUTION PROVEN only after the root
search finishes or every remaining branch is safely pruned. If interrupted, its
absolute color-count upper bound is reported as a conservative gap.

## Logging and configuration

The controller writes JSON Lines session events containing captures, confidence,
recommendations, search statistics, expected/actual boards, and safety errors.
Use config.example.toml as a starting point. Important settings include
patch_ratio, confidence_threshold, settle frame count/interval, solver time,
beam width, node limit, and click delay.

## Known limitations

- Browser zoom, moving/resizing the window, and compositor scaling can invalidate
  a previously selected region; rescan and recalibrate after geometry changes.
- Transparent global windows are compositor-dependent under Wayland. The
  companion debug view is the deliberate fallback.
- ydotool requires a working uinput daemon and appropriate permissions.
- No OCR is required or used for score tracking; the application tracks the
  simulated square scores. Displayed-score OCR can be added later as a check.
- The absolute color-count bound is safe but loose. Exact search is intended for
  small boards; beam/rollout are the practical modes for a 10 x 10 board.
