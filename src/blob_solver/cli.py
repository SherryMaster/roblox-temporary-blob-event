"""Command-line tools for board solving, single-scan automation, and diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import monotonic

from blob_solver.app.config import load_config, save_config
from blob_solver.app.controller import AutomationController, VisionSafetyError
from blob_solver.desktop.coordinates import CoordinateError, CoordinateMapper, query_hyprland_monitors
from blob_solver.desktop.factory import create_desktop_backend
from blob_solver.desktop.overlay import HighlightSpec
from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups
from blob_solver.game.transition import apply_move
from blob_solver.solver.base import Solution
from blob_solver.solver.beam import BeamConfig, BeamSolver, LegacyBeamSolver
from blob_solver.solver.exact import ExactSolver
from blob_solver.solver.greedy import GreedySolver
from blob_solver.solver.hybrid import HybridConfig, HybridPlanner, PRESETS
from blob_solver.solver.plan import Plan
from blob_solver.solver.rollout import RolloutConfig, RolloutSolver
from blob_solver.vision.debug import format_observation, render_observation
from blob_solver.vision.region import Region


def parse_board_lines(lines: list[str]) -> Board:
    """Parse rows top-to-bottom; whitespace-separated tokens are also accepted."""

    rows: list[list[str | None]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split() if " " in line or "\t" in line else list(line)
        rows.append([None if token in {".", "_", "-"} else token for token in tokens])
    if not rows:
        raise ValueError("board file contains no rows")
    return Board.from_matrix(rows, empty=None, strict=True)


def parse_board_file(path: str | Path) -> Board:
    return parse_board_lines(Path(path).read_text(encoding="utf-8").splitlines())


def format_board(board: Board, *, rows: int | None = None, cols: int | None = None) -> str:
    matrix = board.to_matrix(rows=rows, cols=cols, empty=".")
    return "\n".join(" ".join(str(value) for value in row) for row in matrix)


def format_plan(plan: Plan) -> str:
    stats = dict(plan.solver_stats)
    lines = [
        plan.status,
        f"Projected total: {plan.total_score}",
        f"Moves: {plan.move_count}",
        f"Blocks cleared: {plan.blocks_cleared} / {plan.initial_board.block_count}",
        f"Leftover blocks: {plan.leftover_blocks}",
        f"Search time: {float(stats.get('search_time_seconds', 0.0)):.3f}s",
        f"States examined: {int(stats.get('states_examined', stats.get('nodes', 0)))}",
        f"Terminal plans evaluated: {int(stats.get('terminal_plans', 0))}",
        f"Upper bound: {plan.upper_bound}",
        f"Gap: <= {plan.gap}",
        f"Optimal proven: {'Yes' if plan.optimal_proven else 'No'}",
    ]
    if plan.steps:
        lines.extend(["", "#   Color       Size  Score  Cumulative", "--  ----------  ----  -----  ----------"])
        lines.extend(
            f"{step.index:2d}  {str(step.move.color):10s}  {step.group_size:4d}  +{step.immediate_score:4d}  {step.cumulative_score:10d}"
            for step in plan.steps
        )
    return "\n".join(lines)


def format_solution(solution: Solution, board: Board | None = None) -> str:
    if board is not None:
        return format_plan(solution.to_plan(board))
    lines = [
        solution.status,
        f"Projected total: {solution.total_score}",
        f"Moves: {solution.move_count}",
        f"Optimal proven: {solution.optimal_proven}",
        f"Search time: {solution.search_time_seconds:.3f}s",
        f"States examined: {solution.nodes_examined}",
        f"Solver: {solution.solver_name}",
    ]
    if solution.upper_bound is not None:
        lines.append(f"Upper bound: {solution.upper_bound} (gap <= {max(0, solution.upper_bound - solution.total_score)})")
    return "\n".join(lines)


def _solver(name: str, *, beam_width: int, max_nodes: int | None, seed: int, min_group: int = 2, quality: str = "balanced"):
    name = name.lower()
    if name in {"greedy", "greedy-score"}:
        return GreedySolver(policy="score", min_group=min_group)
    if name == "greedy-size":
        return GreedySolver(policy="size", min_group=min_group)
    if name == "beam":
        return BeamSolver(BeamConfig(beam_width=beam_width, max_nodes=max_nodes or 250_000, min_group=min_group))
    if name in {"legacy-beam", "old-beam"}:
        return LegacyBeamSolver(BeamConfig(beam_width=beam_width, max_nodes=max_nodes or 250_000, min_group=min_group))
    if name == "exact":
        return ExactSolver(max_nodes=max_nodes, min_group=min_group)
    if name in {"rollout", "random"}:
        return RolloutSolver(RolloutConfig(seed=seed, min_group=min_group))
    if name in {"hybrid", "fast", "balanced", "deep", "exhaustive"}:
        return HybridPlanner(HybridConfig(quality=name if name in PRESETS else quality, beam_width=beam_width, max_nodes=max_nodes, seed=seed, min_group=min_group))
    raise ValueError(f"unknown solver {name}")


def cmd_solve_board(args: argparse.Namespace) -> int:
    board = parse_board_file(args.board)
    print("Observed board:")
    print(format_board(board))
    solver = _solver(
        args.solver,
        beam_width=args.beam_width,
        max_nodes=args.max_nodes,
        seed=args.seed,
        min_group=args.min_group,
        quality=args.quality,
    )
    solution = solver.solve(board, time_limit=args.time)
    print("\n" + format_solution(solution, board))
    return 0


def cmd_inspect_board(args: argparse.Namespace) -> int:
    board = parse_board_file(args.board)
    print(format_board(board))
    legal = find_groups(board)
    print(f"\nLegal groups: {len(legal)}")
    for index, move in enumerate(legal):
        print(f"{index}: color={move.color} size={move.size} score=+{move.immediate_score} cells={move.cells}")
    if args.move is not None:
        if not 0 <= args.move < len(legal):
            raise ValueError("move index is outside the legal-group list")
        result = apply_move(board, legal[args.move])
        print(f"\nAfter move {args.move}:")
        print(format_board(result))
    return 0


def _controller(args: argparse.Namespace) -> AutomationController:
    config = load_config(args.config)
    if getattr(args, "region", None):
        config.region = Region.parse(args.region)
    if getattr(args, "rows", None) is not None:
        config.board.rows = args.rows
    if getattr(args, "cols", None) is not None:
        config.board.cols = args.cols
    if getattr(args, "backend", None):
        config.desktop.backend = args.backend
    desktop = create_desktop_backend(config.desktop.backend)
    controller = AutomationController(config, config_path=args.config, desktop=desktop)
    if config.region is None:
        controller.select_region()
    return controller


def cmd_select_region(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.backend:
        config.desktop.backend = args.backend
    desktop = create_desktop_backend(config.desktop.backend)
    region = desktop.selector.select_region()
    config.region = region
    save_config(config, args.config)
    print(f"Saved region: {region.x},{region.y},{region.width},{region.height}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    controller = _controller(args)
    observation = controller.scan()
    print(format_observation(observation))
    if args.output:
        render_observation(observation).save(args.output)
        print(f"\nDebug preview: {args.output}")
    return 0


def cmd_solve_capture(args: argparse.Namespace) -> int:
    controller = _controller(args)
    plan = controller.scan_and_solve(quality=args.quality, time_limit=args.time)
    if controller.current_observation is not None:
        print(format_observation(controller.current_observation))
    print("\n" + format_plan(plan))
    return 0


def cmd_hint(args: argparse.Namespace) -> int:
    controller = _controller(args)
    plan = controller.scan_and_solve(quality=args.quality, time_limit=args.time)
    spec = controller.show_step()
    print(format_plan(plan))
    if spec is None:
        print("\nNo legal moves remain.")
        return 0
    print(f"\nStep {spec.step} / {spec.step_count}: click the highlighted {spec.color} group")
    if not args.once:
        response = input("Perform the highlighted move, then press Enter (q to stop): ")
        if response.strip().lower() != "q":
            controller.next_step()
            print("Advanced using the frozen predicted board; no capture was performed.")
    return 0


def cmd_autoplay(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise ValueError("autoplay requires --confirm; this is a deliberate safety gate")
    controller = _controller(args)
    if args.verified:
        controller.config.automation.verified_execution = True
    controller.scan_and_solve(quality=args.quality, time_limit=args.time)
    moves = controller.run_autoplay()
    if args.verified:
        print(f"Verified autoplay finished after {moves} moves; readback capture was enabled explicitly.")
    else:
        print(f"Autoplay finished after {moves} frozen steps; no normal readback captures were used.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    board = parse_board_file(args.board)
    names = ["greedy", "old-beam", "rollout", "hybrid-fast", "hybrid-balanced", "hybrid-deep", "exact"]
    results: list[tuple[str, Solution, float]] = []
    for name in names:
        solver_name = name.split("-", 1)[1] if name.startswith("hybrid-") else name
        started = monotonic()
        solver = _solver(
            solver_name,
            beam_width=args.beam_width,
            max_nodes=args.max_nodes,
            seed=args.seed,
            min_group=args.min_group,
        )
        solution = solver.solve(board, time_limit=args.time)
        results.append((name, solution, monotonic() - started))
    exact = next((solution for name, solution, _ in results if name == "exact"), None)
    exact_score = exact.total_score if exact and exact.optimal_proven else None
    print("solver\tscore\tgap_to_exact\tmoves\tstates\ttime\tstatus")
    print("------\t-----\t------------\t-----\t------\t----\t------")
    for name, solution, elapsed in results:
        gap = "-" if exact_score is None or name == "exact" else str(max(0, exact_score - solution.total_score))
        print(f"{name}\t{solution.total_score}\t{gap}\t{solution.move_count}\t{solution.nodes_examined}\t{elapsed:.3f}s\t{solution.status}")
    return 0


def cmd_capture_fixture(args: argparse.Namespace) -> int:
    controller = _controller(args)
    observation = controller.scan()
    generation = controller.session.generation
    if generation is None:
        raise VisionSafetyError("capture did not produce a frozen screenshot")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generation.screenshot.save(output)
    matrix_path = Path(args.matrix_output) if args.matrix_output else output.with_suffix(".txt")
    matrix_path.write_text(observation.text() + "\n", encoding="utf-8")
    print(f"Saved real capture fixture: {output}")
    print(f"Saved expected symbolic matrix: {matrix_path}")
    return 0


def cmd_overlay_test(args: argparse.Namespace) -> int:
    controller = _controller(args)
    region = controller._require_region()
    rows, cols = controller.config.board.rows, controller.config.board.cols
    cells = tuple(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if row == col or row + col == cols - 1
    )
    spec = HighlightSpec(
        region=region,
        rows=rows,
        cols=cols,
        cells=cells,
        click_cell=(0, 0),
        group_size=len(cells),
        immediate_score=0,
        projected_total=0,
        step=1,
        step_count=1,
        color="diagnostic",
    )
    try:
        geometry = CoordinateMapper(query_hyprland_monitors()).overlay_geometry(region)
        print(f"monitor={geometry.monitor.name} scale={geometry.monitor.scale}")
        print(f"capture region={region.to_dict()}")
        print(f"logical overlay region={geometry.local_region.to_dict()}")
        diagnostic = CoordinateMapper((geometry.monitor,)).click_diagnostic(
            region,
            rows=rows,
            cols=cols,
            screen_row=rows - 1,
            col=0,
        )
        print(f"calculated click point={diagnostic.input_click_point} ({diagnostic.input_space})")
    except CoordinateError as exc:
        print(f"coordinate diagnostics unavailable: {exc}")
    controller.desktop.overlay.show(spec)
    print("Layer-shell diagnostic is visible. Click through a highlighted cell, then press Enter.")
    try:
        input()
    finally:
        controller.desktop.overlay.close()
    return 0


def cmd_board_debug(args: argparse.Namespace) -> int:
    if args.board:
        board = parse_board_file(args.board)
    elif not sys.stdin.isatty():
        board = parse_board_lines(sys.stdin.read().splitlines())
    else:
        print("Paste rows top-to-bottom; enter a blank line when finished.")
        rows: list[str] = []
        while True:
            line = input()
            if not line.strip():
                break
            rows.append(line)
        board = parse_board_lines(rows)
    while True:
        print("\nCurrent board:")
        print(format_board(board))
        legal = find_groups(board)
        print(f"Legal groups: {len(legal)}")
        for index, move in enumerate(legal):
            print(f"{index}: {move.color} size={move.size} +{move.immediate_score} cells={move.cells}")
        if not legal or args.once:
            return 0
        choice = input("Move index to apply, or q to quit: ").strip().lower()
        if choice == "q":
            return 0
        try:
            board = apply_move(board, legal[int(choice)])
        except (ValueError, IndexError) as exc:
            print(f"Invalid move selection: {exc}")


def cmd_ui(args: argparse.Namespace) -> int:
    from blob_solver.app.ui import run_app

    return run_app(config_path=args.config)


def _add_capture_options(command: argparse.ArgumentParser, *, include_output: bool = True) -> None:
    command.add_argument("--region", default=None, help="x,y,width,height")
    command.add_argument("--rows", type=int, default=None)
    command.add_argument("--cols", type=int, default=None)
    command.add_argument("--backend", choices=["auto", "wayland", "x11"], default=None)
    if include_output:
        command.add_argument("--output", default=None, help="vision-debug output path")
    command.add_argument("--quality", choices=["fast", "balanced", "deep", "exhaustive"], default="balanced")
    command.add_argument(
        "--time",
        type=float,
        default=None,
        help="wall-clock override; omit to use the quality preset",
    )
    command.add_argument("--once", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blob-solver",
        description="SameGame vision planner and Wayland assistant",
    )
    parser.add_argument("--config", default=None, help="TOML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-region", help="select and save the board rectangle")
    select.add_argument("--backend", choices=["auto", "wayland", "x11"], default=None)
    select.set_defaults(func=cmd_select_region)

    board = subparsers.add_parser("solve-board", help="solve a textual board without screenshots")
    board.add_argument("board", type=Path)
    board.add_argument(
        "--solver",
        choices=["hybrid", "fast", "balanced", "deep", "exhaustive", "greedy", "greedy-size", "beam", "legacy-beam", "rollout", "exact"],
        default="hybrid",
    )
    board.add_argument("--quality", choices=["fast", "balanced", "deep", "exhaustive"], default="balanced")
    board.add_argument(
        "--time",
        type=float,
        default=None,
        help="wall-clock override; omit to use the quality preset",
    )
    board.add_argument("--beam-width", type=int, default=700)
    board.add_argument("--max-nodes", type=int, default=180_000)
    board.add_argument("--seed", type=int, default=0)
    board.add_argument("--min-group", type=int, default=2)
    board.set_defaults(func=cmd_solve_board)

    inspect = subparsers.add_parser("inspect-board", help="print legal groups and optionally apply one")
    inspect.add_argument("board", type=Path)
    inspect.add_argument("--move", type=int, default=None)
    inspect.set_defaults(func=cmd_inspect_board)

    benchmark = subparsers.add_parser("benchmark", help="compare baseline, search, hybrid, and exact modes")
    benchmark.add_argument("board", type=Path)
    benchmark.add_argument("--time", type=float, default=2.0)
    benchmark.add_argument("--beam-width", type=int, default=500)
    benchmark.add_argument("--max-nodes", type=int, default=100_000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--min-group", type=int, default=2)
    benchmark.set_defaults(func=cmd_benchmark)

    board_debug = subparsers.add_parser("board-debug", help="paste or read a symbolic board and apply moves")
    board_debug.add_argument("board", type=Path, nargs="?", default=None)
    board_debug.add_argument("--once", action="store_true")
    board_debug.set_defaults(func=cmd_board_debug)

    scan = subparsers.add_parser("scan", help="capture once and print the reconstructed board")
    _add_capture_options(scan)
    scan.set_defaults(func=cmd_scan)

    vision_debug = subparsers.add_parser("vision-debug", help="capture once and save a labeled preview")
    _add_capture_options(vision_debug)
    vision_debug.set_defaults(func=cmd_scan)

    solve = subparsers.add_parser("solve", help="capture once and solve a complete frozen generation")
    _add_capture_options(solve)
    solve.set_defaults(func=cmd_solve_capture)

    hint = subparsers.add_parser("hint", help="show frozen plan steps without automatic rescans")
    _add_capture_options(hint)
    hint.set_defaults(func=cmd_hint)

    autoplay = subparsers.add_parser("autoplay", help="execute the frozen plan; Verified mode is opt-in")
    _add_capture_options(autoplay)
    autoplay.add_argument("--confirm", action="store_true")
    autoplay.add_argument("--verified", action="store_true", help="explicitly enable capture-after-each-move verification")
    autoplay.set_defaults(func=cmd_autoplay)

    fixture = subparsers.add_parser("capture-fixture", help="save one real capture and its symbolic matrix")
    _add_capture_options(fixture, include_output=False)
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--matrix-output", default=None)
    fixture.set_defaults(func=cmd_capture_fixture)

    overlay = subparsers.add_parser("overlay-test", help="diagnose and visually test the native Wayland layer overlay")
    _add_capture_options(overlay)
    overlay.set_defaults(func=cmd_overlay_test)

    ui = subparsers.add_parser("ui", help="open the PySide6 control panel")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, VisionSafetyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
