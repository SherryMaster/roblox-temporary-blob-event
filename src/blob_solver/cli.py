"""Command-line tools for board solving, vision debugging, and automation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import monotonic

from blob_solver.app.config import load_config, save_config
from blob_solver.app.controller import AutomationController, VisionSafetyError
from blob_solver.desktop.factory import create_desktop_backend
from blob_solver.game.board import Board
from blob_solver.game.groups import find_groups
from blob_solver.game.transition import apply_move
from blob_solver.solver.base import Solution
from blob_solver.solver.beam import BeamConfig, BeamSolver
from blob_solver.solver.exact import ExactSolver
from blob_solver.solver.greedy import GreedySolver
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


def format_solution(solution: Solution) -> str:
    lines = [
        f"Total additional score: {solution.total_score}",
        f"Moves: {solution.move_count}",
        f"Optimal proven: {solution.optimal_proven}",
        f"Search time: {solution.search_time_seconds:.3f} sec",
        f"Nodes examined: {solution.nodes_examined}",
        f"Solver: {solution.solver_name}",
        f"Status: {solution.status}",
    ]
    if solution.upper_bound is not None:
        gap = max(0, solution.upper_bound - solution.total_score)
        lines.append(f"Upper bound: {solution.upper_bound} (gap {gap})")
    for index, move in enumerate(solution.moves, start=1):
        lines.append(
            f"{index:3d}. {move.color!s:8s} size {move.size:3d} "
            f"+{move.immediate_score:4d} click={move.click_cell}"
        )
    return "\n".join(lines)


def _solver(
    name: str,
    *,
    beam_width: int,
    max_nodes: int,
    seed: int,
    min_group: int = 2,
):
    name = name.lower()
    if name in {"greedy", "greedy-score"}:
        return GreedySolver(policy="score", min_group=min_group)
    if name == "greedy-size":
        return GreedySolver(policy="size", min_group=min_group)
    if name == "beam":
        return BeamSolver(
            BeamConfig(beam_width=beam_width, max_nodes=max_nodes, min_group=min_group)
        )
    if name == "exact":
        return ExactSolver(max_nodes=max_nodes, min_group=min_group)
    if name in {"rollout", "random"}:
        return RolloutSolver(RolloutConfig(seed=seed, min_group=min_group))
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
    )
    solution = solver.solve(board, time_limit=args.time)
    print("\n" + format_solution(solution))
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
    if args.region:
        config.region = Region.parse(args.region)
    if args.rows is not None:
        config.board.rows = args.rows
    if args.cols is not None:
        config.board.cols = args.cols
    if args.backend:
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


def _ensure_calibrated(controller: AutomationController) -> None:
    if controller.profile is None:
        controller.calibrate()


def cmd_scan(args: argparse.Namespace) -> int:
    controller = _controller(args)
    _ensure_calibrated(controller)
    observation = controller.scan()
    print(format_observation(observation))
    if args.output:
        render_observation(observation).save(args.output)
        print(f"\nDebug preview: {args.output}")
    return 0


def cmd_hint(args: argparse.Namespace) -> int:
    controller = _controller(args)
    _ensure_calibrated(controller)
    observation = controller.scan()
    print(format_observation(observation))
    solution = controller.analyze(mode=args.solver, time_limit=args.time)
    spec = controller.show_next_move(solution)
    print("\n" + format_solution(solution))
    if spec is None:
        print("\nNo legal moves remain.")
        return 0
    print(f"\nHighlighted click cell: {spec.click_cell}")
    if not args.once:
        response = input("\nPerform the highlighted move, then press Enter (q to stop): ")
        if response.strip().lower() != "q":
            action = controller.recheck_manual_move()
            if action is not None:
                print(f"Readback matched prediction: {action.matched}")
    return 0


def cmd_solve_capture(args: argparse.Namespace) -> int:
    controller = _controller(args)
    _ensure_calibrated(controller)
    observation = controller.scan()
    solution = controller.analyze(mode=args.solver, time_limit=args.time)
    print(format_observation(observation))
    print("\n" + format_solution(solution))
    return 0


def cmd_autoplay(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise ValueError("autoplay requires --confirm; this is a deliberate safety gate")
    controller = _controller(args)
    _ensure_calibrated(controller)
    controller.scan()
    moves = controller.run_autoplay()
    print(f"Autoplay finished after {moves} verified moves; estimated score={controller.estimated_score}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    board = parse_board_file(args.board)
    names = ["greedy", "beam", "rollout", "exact"]
    results: list[tuple[str, Solution, float]] = []
    for name in names:
        started = monotonic()
        solver = _solver(
            name,
            beam_width=args.beam_width,
            max_nodes=args.max_nodes,
            seed=args.seed,
            min_group=args.min_group,
        )
        solution = solver.solve(board, time_limit=args.time)
        elapsed = monotonic() - started
        results.append((name, solution, elapsed))
    exact = next((solution for name, solution, _ in results if name == "exact"), None)
    exact_score = exact.total_score if exact and exact.optimal_proven else None
    print("solver\tscore\tgap_to_exact\tnodes\ttime\tstatus")
    print("------\t-----\t------------\t-----\t----\t------")
    for name, solution, elapsed in results:
        gap = "-" if exact_score is None or name == "exact" else str(max(0, exact_score - solution.total_score))
        print(
            f"{name}\t{solution.total_score}\t{gap}\t{solution.nodes_examined}"
            f"\t{elapsed:.3f}s\t{solution.status}"
        )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blob-solver",
        description="SameGame vision solver and safe desktop assistant",
    )
    parser.add_argument("--config", default=None, help="TOML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-region", help="select and save the board rectangle")
    select.add_argument("--backend", choices=["auto", "wayland", "x11"], default=None)
    select.set_defaults(func=cmd_select_region)

    board = subparsers.add_parser("solve-board", help="solve a textual board without screenshots")
    board.add_argument("board", type=Path)
    board.add_argument("--solver", choices=["greedy", "greedy-size", "beam", "rollout", "exact"], default="beam")
    board.add_argument("--time", type=float, default=3.0)
    board.add_argument("--beam-width", type=int, default=1000)
    board.add_argument("--max-nodes", type=int, default=250000)
    board.add_argument("--seed", type=int, default=0)
    board.add_argument("--min-group", type=int, default=2)
    board.set_defaults(func=cmd_solve_board)

    inspect = subparsers.add_parser("inspect-board", help="print legal groups and optionally apply one")
    inspect.add_argument("board", type=Path)
    inspect.add_argument("--move", type=int, default=None)
    inspect.set_defaults(func=cmd_inspect_board)

    benchmark = subparsers.add_parser("benchmark", help="compare solver modes on a textual board")
    benchmark.add_argument("board", type=Path)
    benchmark.add_argument("--time", type=float, default=3.0)
    benchmark.add_argument("--beam-width", type=int, default=1000)
    benchmark.add_argument("--max-nodes", type=int, default=250000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--min-group", type=int, default=2)
    benchmark.set_defaults(func=cmd_benchmark)

    board_debug = subparsers.add_parser(
        "board-debug",
        help="paste or read a symbolic board and interactively apply moves",
    )
    board_debug.add_argument("board", type=Path, nargs="?")
    board_debug.add_argument("--once", action="store_true")
    board_debug.set_defaults(func=cmd_board_debug)

    for name, help_text in (
        ("scan", "capture and print the reconstructed board"),
        ("vision-debug", "capture, print, and optionally save a labeled preview"),
        ("solve", "capture the board and print a complete recommended plan"),
        ("hint", "show one recommended move and optionally validate a manual click"),
        ("autoplay", "perform verified clicks until terminal"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--region", default=None, help="x,y,width,height")
        command.add_argument("--rows", type=int, default=None)
        command.add_argument("--cols", type=int, default=None)
        command.add_argument("--backend", choices=["auto", "wayland", "x11"], default=None)
        command.add_argument("--output", default=None, help="vision-debug image output path")
        command.add_argument("--solver", choices=["greedy", "greedy-size", "beam", "rollout", "exact"], default=None)
        command.add_argument("--time", type=float, default=3.0)
        command.add_argument("--once", action="store_true")
        command.set_defaults(
            func={
                "scan": cmd_scan,
                "vision-debug": cmd_scan,
                "solve": cmd_solve_capture,
                "hint": cmd_hint,
                "autoplay": cmd_autoplay,
            }[name]
        )
        if name == "autoplay":
            command.add_argument("--confirm", action="store_true")

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
