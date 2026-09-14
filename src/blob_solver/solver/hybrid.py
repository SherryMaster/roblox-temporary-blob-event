"""Anytime portfolio planning for complete deterministic SameGame generations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import __main__
import multiprocessing as mp
import os
from pathlib import Path
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from threading import Event
from time import monotonic
from typing import Callable

from blob_solver.game.board import Board
from blob_solver.game.groups import Move, find_groups
from blob_solver.game.rules import GameRules
from blob_solver.game.transition import apply_move

from .base import SearchBudget, SearchStats, Solution, finish_solution, score_transition
from .exact import ExactSolver
from .greedy import GreedySolver
from .heuristics import color_ideal_upper_bound, component_features, evaluate_board, ordered_moves
from .plan import RootMoveEvaluation


@dataclass(frozen=True, slots=True)
class HybridPreset:
    name: str
    default_time_limit: float | None
    beam_width: int
    max_nodes: int | None
    candidate_moves: int | None
    rollout_count: int
    exact_endgame_blocks: int
    exact_legal_groups: int
    exact_work_limit: int


PRESETS: dict[str, HybridPreset] = {
    "fast": HybridPreset("Fast", 1.0, 240, 40_000, 24, 24, 18, 6, 90),
    "balanced": HybridPreset("Balanced", 5.0, 700, 180_000, 48, 80, 25, 8, 140),
    "deep": HybridPreset("Deep", 20.0, 1_500, 650_000, 80, 220, 31, 10, 220),
    "exhaustive": HybridPreset("Exhaustive", None, 3_000, None, None, 0, 36, 12, 320),
}


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Search controls. Quality presets are the normal user-facing choice."""

    quality: str = "balanced"
    beam_width: int | None = None
    max_nodes: int | None = None
    candidate_moves: int | None = None
    rollout_count: int | None = None
    exact_endgame_blocks: int | None = None
    exact_legal_groups: int | None = None
    exact_work_limit: int | None = None
    seed: int = 0
    min_group: int = 2
    process_count: int = 0

    def __post_init__(self) -> None:
        quality = self.quality.lower()
        if quality not in PRESETS:
            raise ValueError(f"unknown hybrid quality: {self.quality}")
        if self.beam_width is not None and self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("max_nodes must be positive or None")
        if self.candidate_moves is not None and self.candidate_moves < 1:
            raise ValueError("candidate_moves must be positive or None")
        if self.rollout_count is not None and self.rollout_count < 0:
            raise ValueError("rollout_count must not be negative")
        if self.min_group < 2:
            raise ValueError("min_group must be at least 2")
        if self.process_count < 0:
            raise ValueError("process_count must be zero (automatic) or positive")

    @property
    def preset(self) -> HybridPreset:
        return PRESETS[self.quality.lower()]

    def value(self, name: str):
        explicit = getattr(self, name)
        return getattr(self.preset, name) if explicit is None else explicit


@dataclass(frozen=True, slots=True)
class SearchProgress:
    phase: str
    best_complete_score: int
    upper_bound: int
    states_examined: int
    terminal_plans: int
    elapsed_seconds: float
    done: bool = False


@dataclass(frozen=True, slots=True)
class _SearchNode:
    board: Board
    score_so_far: int
    path: tuple[Move, ...]
    lower_bound_total: int
    upper_bound_total: int
    completion: tuple[Move, ...]
    completion_score: int
    root_index: int
    topology_key: tuple[int, ...]


def _root_rollout_worker(
    payload: tuple[int, Board, Move, HybridConfig, GameRules],
) -> tuple[int, Move, tuple[Move, ...], int, int]:
    """Explore one independent root branch without importing desktop/UI code."""

    root_index, board, first_move, config, rules = payload
    planner = HybridPlanner(config, rules=rules)
    stats = SearchStats()
    budget = SearchBudget(time_limit=None, cancel_event=None, max_nodes=None, stats=stats)
    next_board = apply_move(board, first_move, min_group=config.min_group, validate=False)
    suffix = planner._complete_suffix(
        next_board,
        rng=random.Random(config.seed + 20_000 + root_index),
        policy="score",
        budget=budget,
        bounded=False,
        stats=stats,
        exact_cache={},
        allow_exact=False,
    )
    if suffix is None:  # pragma: no cover - the unbounded worker normally completes
        return root_index, first_move, (), 0, stats.states_examined
    suffix_moves, suffix_score, _ = suffix
    return root_index, first_move, suffix_moves, suffix_score, stats.states_examined


class HybridPlanner:
    """Search complete continuations while retaining an anytime incumbent.

    The planner deliberately has one product-facing entry point. Its internal
    portfolio combines diverse complete rollouts, fair root exploration,
    best-first upper-bound search, exact late-game suffixes, and randomized
    restarts. Every candidate used as an incumbent is a terminal replay.
    """

    def __init__(
        self,
        config: HybridConfig | None = None,
        *,
        rules: GameRules | None = None,
    ) -> None:
        self.config = config or HybridConfig()
        self.rules = rules or GameRules(min_group=self.config.min_group)
        self.last_root_evaluations: tuple[RootMoveEvaluation, ...] = ()

    def _choose_move(
        self,
        board: Board,
        legal: tuple[Move, ...],
        policy: str,
        rng: random.Random,
    ) -> Move:
        if policy == "score":
            return max(legal, key=lambda move: (move.immediate_score, move.size, move.cells))
        if policy == "size":
            return max(legal, key=lambda move: (move.size, move.immediate_score, move.cells))

        if policy == "topology":
            # Preserve a useful consolidation lookahead without evaluating a
            # full one-ply feature tree at every state of an instant rollout.
            # Immediate score is only a prefilter; the final choice still uses
            # topology/merge features among these candidates.
            topology_candidates = tuple(
                sorted(
                    legal,
                    key=lambda move: (move.immediate_score, move.size, move.cells),
                    reverse=True,
                )[:6]
            )
            return ordered_moves(board, topology_candidates, min_group=self.config.min_group)[0]

        if policy == "small":
            scored: list[tuple[tuple[float, ...], Move]] = []
            for move in legal:
                after = apply_move(board, move, validate=False)
                features = component_features(after, min_group=self.config.min_group)
                scored.append(
                    (
                        (
                            -float(move.size),
                            features["near_merge_gain"],
                            features["merge_gain"],
                            -features["blocker_cells"],
                            -float(move.immediate_score),
                        ),
                        move,
                    )
                )
            return max(scored, key=lambda item: (item[0], item[1].cells))[1]

        if policy == "low-fragmentation":
            return max(
                legal,
                key=lambda move: (
                    evaluate_board(
                        apply_move(board, move, validate=False),
                        min_group=self.config.min_group,
                    ),
                    move.immediate_score,
                    move.cells,
                ),
            )

        if policy == "merge":
            candidates: list[tuple[float, Move]] = []
            before = component_features(board, min_group=self.config.min_group)
            for move in legal:
                after = apply_move(board, move, validate=False)
                features = component_features(after, min_group=self.config.min_group)
                value = (
                    1.35 * (features["near_merge_gain"] - before["near_merge_gain"])
                    + 0.55 * features["merge_gain"]
                    + 0.25 * features["largest"]
                    + 0.12 * move.immediate_score
                    + 2.0 * (board.width - after.width)
                )
                candidates.append((value, move))
            return max(candidates, key=lambda item: (item[0], item[1].cells))[1]

        if policy == "random":
            candidates = ordered_moves(board, legal, min_group=self.config.min_group)
            candidates = candidates[: min(len(candidates), 12)]
            weights = []
            for move in candidates:
                after = apply_move(board, move, validate=False)
                features = component_features(after, min_group=self.config.min_group)
                weights.append(
                    max(
                        0.01,
                        1.0
                        + move.immediate_score
                        + 0.45 * features["near_merge_gain"]
                        + 0.18 * features["largest"],
                    )
                )
            return rng.choices(candidates, weights=weights, k=1)[0]

        raise ValueError(f"unknown rollout policy: {policy}")

    def _rollout(
        self,
        board: Board,
        *,
        policy: str,
        rng: random.Random,
        budget: SearchBudget,
        bounded: bool,
        stats: SearchStats,
    ) -> tuple[tuple[Move, ...], int] | None:
        current = board
        path: list[Move] = []
        score = 0
        while True:
            if bounded:
                if not budget.visit():
                    return None
            else:
                stats.rollout_nodes += 1
            legal = find_groups(current, min_group=self.config.min_group)
            if not legal:
                stats.terminal_plans += 1
                stats.complete_rollouts += 1
                return tuple(path), score
            move = self._choose_move(current, legal, policy, rng)
            path.append(move)
            score += score_transition(current, move, self.rules)
            current = apply_move(current, move, min_group=self.config.min_group, validate=False)

    def _remaining_seconds(self, budget: SearchBudget) -> float | None:
        if budget.deadline is None:
            return None
        return max(0.0, budget.deadline - monotonic())

    def _exact_eligible(self, board: Board) -> bool:
        groups = find_groups(board, min_group=self.config.min_group)
        blocks = board.block_count
        legal_count = len(groups)
        work = legal_count * max(1, blocks // self.config.min_group)
        return (
            blocks <= self.config.value("exact_endgame_blocks")
            or (
                blocks <= 2 * self.config.value("exact_endgame_blocks")
                and legal_count <= self.config.value("exact_legal_groups")
            )
            or work <= self.config.value("exact_work_limit")
        )

    def _complete_suffix(
        self,
        board: Board,
        *,
        rng: random.Random,
        policy: str = "score",
        budget: SearchBudget,
        bounded: bool,
        stats: SearchStats,
        exact_cache: dict[Board, tuple[tuple[Move, ...], int]],
        allow_exact: bool = True,
    ) -> tuple[tuple[Move, ...], int, bool] | None:
        if budget.expired():
            return None
        cached = exact_cache.get(board)
        if cached is not None:
            stats.complete_rollouts += 1
            return cached[0], cached[1], True
        if allow_exact and self._exact_eligible(board):
            remaining = self._remaining_seconds(budget)
            exact = ExactSolver(
                max_nodes=self.config.value("max_nodes"),
                min_group=self.config.min_group,
                rules=self.rules,
            ).solve(
                board,
                time_limit=remaining,
                cancel_event=budget.cancel_event,
            )
            stats.nodes += exact.nodes_examined
            stats.terminal_plans += exact.terminal_plans
            stats.complete_rollouts += 1
            if exact.optimal_proven:
                exact_cache[board] = (exact.moves, exact.total_score)
            return exact.moves, exact.total_score, exact.optimal_proven
        rollout = self._rollout(
            board,
            policy=policy,
            rng=rng,
            budget=budget,
            bounded=bounded,
            stats=stats,
        )
        if rollout is None:
            return None
        return rollout[0], rollout[1], False

    @staticmethod
    def _topology_key(board: Board, root_index: int, min_group: int) -> tuple[int, ...]:
        legal = find_groups(board, min_group=min_group)
        largest = max((move.size for move in legal), default=0)
        return (root_index, board.width, board.block_count // 5, len(legal), largest)

    def _trim_frontier(
        self,
        entries: list[tuple[tuple[object, ...], int, _SearchNode]],
        *,
        limit: int,
    ) -> list[tuple[tuple[object, ...], int, _SearchNode]]:
        if len(entries) <= limit:
            return entries
        entries.sort(key=lambda entry: entry[0])
        roots: dict[int, list[tuple[tuple[object, ...], int, _SearchNode]]] = {}
        for entry in entries:
            roots.setdefault(entry[2].root_index, []).append(entry)
        root_count = max(1, len(roots))
        per_root = max(1, limit // root_count)
        selected: list[tuple[tuple[object, ...], int, _SearchNode]] = []
        for root_entries in roots.values():
            selected.extend(root_entries[:per_root])
        selected_ids = {id(entry[2]) for entry in selected}
        for entry in entries:
            if len(selected) >= limit:
                break
            if id(entry[2]) not in selected_ids:
                selected.append(entry)
                selected_ids.add(id(entry[2]))
        selected = selected[:limit]
        heapq.heapify(selected)
        return selected

    def _root_worker_count(self, root_count: int) -> int:
        requested = self.config.process_count
        if requested == 1 or root_count < 8:
            return 1
        # forkserver/spawn needs an importable main module. Interactive
        # ``python -c``/stdin callers are developer conveniences, not a safe
        # process-pool launch context; keep them serial and avoid noisy child
        # tracebacks before falling back.
        main_file = getattr(__main__, "__file__", None)
        if not main_file or not Path(main_file).exists():
            return 1
        available = os.cpu_count() or 1
        if requested <= 0:
            requested = min(4, available)
        return min(max(1, requested), root_count, available)

    def _root_rollouts(
        self,
        board: Board,
        root_moves: tuple[Move, ...],
        *,
        budget: SearchBudget,
    ) -> list[tuple[int, Move, tuple[Move, ...], int, int]]:
        jobs = [
            (index, board, move, self.config, self.rules)
            for index, move in enumerate(root_moves)
        ]

        def sequential() -> list[tuple[int, Move, tuple[Move, ...], int, int]]:
            results: list[tuple[int, Move, tuple[Move, ...], int, int]] = []
            for job in jobs:
                if budget.expired():
                    break
                results.append(_root_rollout_worker(job))
            return results

        workers = self._root_worker_count(len(root_moves))
        if workers <= 1 or budget.expired():
            return sequential()

        # A process pool is useful only for the independent root continuations.
        # The workers receive immutable Board/Move/config data and return plain
        # tuples; no GUI, socket, or controller state crosses the boundary.
        try:
            methods = mp.get_all_start_methods()
            method = "forkserver" if "forkserver" in methods else "spawn"
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context(method),
            )
        except (OSError, RuntimeError):
            return sequential()

        results: list[tuple[int, Move, tuple[Move, ...], int, int]] = []
        pending = set()
        fallback = False
        try:
            pending = {executor.submit(_root_rollout_worker, job) for job in jobs}
            while pending:
                timeout = 0.05
                if budget.deadline is not None:
                    timeout = min(timeout, max(0.0, budget.deadline - monotonic()))
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                if budget.expired():
                    break
        except Exception:
            # A platform without a usable multiprocessing start path should
            # still get the same complete planner through the serial route.
            fallback = not budget.expired()
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(
                wait=not fallback and not budget.expired(),
                cancel_futures=budget.expired(),
            )
        if fallback:
            return sequential()
        return sorted(results, key=lambda item: item[0])

    def solve(
        self,
        board: Board,
        *,
        time_limit: float | None = None,
        cancel_event: Event | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
    ) -> Solution:
        stats = SearchStats()
        selected_limit = self.config.preset.default_time_limit if time_limit is None else time_limit
        budget = SearchBudget(
            time_limit=selected_limit,
            cancel_event=cancel_event,
            max_nodes=self.config.value("max_nodes"),
            stats=stats,
        )
        rng = random.Random(self.config.seed)
        exact_cache: dict[Board, tuple[tuple[Move, ...], int]] = {}
        best_moves: tuple[Move, ...] = ()
        best_score = -1
        phase = "incumbents"
        global_upper = color_ideal_upper_bound(board, self.rules)
        root_evaluations: dict[int, RootMoveEvaluation] = {}
        root_best: dict[int, tuple[int, tuple[Move, ...]]] = {}
        root_search_nodes: dict[int, int] = {}

        def current_upper() -> int:
            return global_upper

        def emit(*, done: bool = False, force: bool = False) -> None:
            if progress_callback is None and not force:
                return
            snapshot = SearchProgress(
                phase=phase,
                best_complete_score=max(0, best_score),
                upper_bound=current_upper(),
                states_examined=stats.states_examined,
                terminal_plans=stats.terminal_plans,
                elapsed_seconds=stats.elapsed,
                done=done,
            )
            if progress_callback is not None:
                progress_callback(snapshot)

        def consider(
            moves: tuple[Move, ...],
            score: int,
            *,
            root_index: int | None = None,
        ) -> None:
            nonlocal best_moves, best_score
            if root_index is not None:
                previous = root_best.get(root_index)
                if previous is None or score > previous[0] or (
                    score == previous[0] and len(moves) < len(previous[1])
                ):
                    root_best[root_index] = (score, moves)
            if score < best_score:
                return
            if score == best_score and best_moves and len(moves) >= len(best_moves):
                return
            best_score = score
            best_moves = moves
            stats.best_complete_score = score
            emit()

        # Seed a complete incumbent before consulting the budget. This is the
        # cancellation/zero-budget safety net; optional diversity policies
        # below are budget-aware and can never delay returning this plan.
        incumbent = GreedySolver(
            min_group=self.config.min_group,
            rules=self.rules,
        ).solve(board)
        stats.complete_rollouts = 1
        stats.terminal_plans = 1
        stats.rollout_nodes = incumbent.nodes_examined
        consider(incumbent.moves, incumbent.total_score)

        # Phase A: several genuinely complete, diverse incumbents. Every
        # optional rollout is bounded; only the seed above is guaranteed to
        # run even when the caller has already cancelled or timed out. The
        # expensive low-fragmentation rollout is reserved for a longer budget
        # so Fast can reach fair root exploration instead of spending its whole
        # second on one heuristic path.
        phase_a_policies = ["score", "topology"]
        if selected_limit is None or selected_limit >= 2.0:
            phase_a_policies.append("low-fragmentation")
        for index, policy in enumerate(phase_a_policies):
            if budget.expired():
                break
            # Topology extraction is intentionally richer than a score-only
            # rollout. Keep a short-deadline run responsive and reserve time
            # for fair root exploration instead of spending the whole budget
            # on one optional incumbent policy.
            reserve = 0.12 if policy == "low-fragmentation" else 0.08
            if index and budget.deadline is not None and budget.deadline - monotonic() < reserve:
                break
            if policy == "random":
                rollout_rng = random.Random(self.config.seed + 10_000 + index)
            else:
                rollout_rng = rng
            completion = self._rollout(
                board,
                policy=policy,
                rng=rollout_rng,
                budget=budget,
                bounded=True,
                stats=stats,
            )
            if completion is not None:
                consider(completion[0], completion[1])
        phase = "root exploration"
        emit()

        # Phase B: every legal root move gets at least one full continuation.
        root_moves = find_groups(board, min_group=self.config.min_group)
        root_upper_bound = max(
            (
                score_transition(board, move, self.rules)
                + color_ideal_upper_bound(
                    apply_move(board, move, min_group=self.config.min_group, validate=False),
                    self.rules,
                )
                for move in root_moves
            ),
            default=0,
        )
        # A first move removes cells before any continuation can score. The
        # maximum over all such branches is a strictly tighter safe bound than
        # the root color-count bound whenever root moves have different sizes.
        global_upper = root_upper_bound
        frontier_entries: list[tuple[tuple[object, ...], int, _SearchNode]] = []
        root_seen: dict[tuple[Board, int], int] = {}
        frontier_counter = 0
        for root_index, first_move, suffix_moves, suffix_score, branch_nodes in self._root_rollouts(
            board,
            root_moves,
            budget=budget,
        ):
            next_board = apply_move(board, first_move, min_group=self.config.min_group, validate=False)
            immediate = score_transition(board, first_move, self.rules)
            stats.rollout_nodes += branch_nodes
            stats.terminal_plans += 1
            stats.complete_rollouts += 1
            complete_plan = (first_move,) + suffix_moves
            complete_score = immediate + suffix_score
            root_evaluations[root_index] = RootMoveEvaluation(
                first_move=first_move,
                best_complete_score_found=complete_score,
                best_complete_plan=complete_plan,
                upper_bound=immediate + color_ideal_upper_bound(next_board, self.rules),
                search_nodes=branch_nodes,
            )
            root_search_nodes[root_index] = branch_nodes
            consider(complete_plan, complete_score, root_index=root_index)
            node = _SearchNode(
                board=next_board,
                score_so_far=immediate,
                path=(first_move,),
                lower_bound_total=complete_score,
                upper_bound_total=immediate + color_ideal_upper_bound(next_board, self.rules),
                completion=suffix_moves,
                completion_score=suffix_score,
                root_index=root_index,
                topology_key=self._topology_key(next_board, root_index, self.config.min_group),
            )
            key = (next_board, root_index)
            root_seen[key] = immediate
            if node.upper_bound_total > best_score or self.config.quality.lower() == "exhaustive":
                priority = (
                    -node.upper_bound_total,
                    -node.lower_bound_total,
                    -evaluate_board(next_board, min_group=self.config.min_group),
                )
                heapq.heappush(frontier_entries, (priority, frontier_counter, node))
                frontier_counter += 1
        self.last_root_evaluations = tuple(
            sorted(root_evaluations.values(), key=lambda item: item.best_complete_score_found, reverse=True)
        )
        emit()

        # Finish the incumbent portfolio after roots have had a fair chance.
        # These policies are still complete terminal rollouts, and their
        # scores are fed back into the corresponding root report when known.
        phase = "diverse rollouts"
        for index, policy in enumerate(("small", "merge", "random")):
            if budget.expired():
                break
            if budget.deadline is not None and budget.deadline - monotonic() < 0.45:
                break
            completion = self._rollout(
                board,
                policy=policy,
                rng=random.Random(self.config.seed + 30_000 + index),
                budget=budget,
                bounded=True,
                stats=stats,
            )
            if completion is not None:
                moves, score = completion
                root_index = None
                if moves:
                    root_index = next(
                        (candidate_index for candidate_index, candidate in enumerate(root_moves) if candidate == moves[0]),
                        None,
                    )
                consider(moves, score, root_index=root_index)
        emit()

        # Phase C/D/E/F: best-first tree search, randomized restarts, exact
        # suffixes, and safe upper-bound pruning share one anytime frontier.
        phase = "tree search"
        expanded = 0
        rollout_budget = self.config.value("rollout_count")
        while frontier_entries and not budget.expired():
            _, _, node = heapq.heappop(frontier_entries)
            if node.upper_bound_total <= best_score:
                continue
            if not budget.visit():
                break
            expanded += 1
            root_search_nodes[node.root_index] = root_search_nodes.get(node.root_index, 0) + 1
            legal = ordered_moves(node.board, min_group=self.config.min_group)
            candidate_limit = self.config.value("candidate_moves")
            if candidate_limit is not None:
                legal = legal[:candidate_limit]
            if not legal:
                consider(node.path, node.score_so_far)
                continue
            for move in legal:
                if budget.expired():
                    break
                next_board = apply_move(node.board, move, min_group=self.config.min_group, validate=False)
                next_score = node.score_so_far + score_transition(node.board, move, self.rules)
                reach_key = (next_board, node.root_index)
                if next_score <= root_seen.get(reach_key, -1):
                    continue
                root_seen[reach_key] = next_score
                suffix_started_at = stats.states_examined
                suffix = self._complete_suffix(
                    next_board,
                    rng=rng,
                    budget=budget,
                    bounded=True,
                    stats=stats,
                    exact_cache=exact_cache,
                )
                root_search_nodes[node.root_index] += stats.states_examined - suffix_started_at
                if suffix is None:
                    break
                suffix_moves, suffix_score, _ = suffix
                path = node.path + (move,)
                complete_score = next_score + suffix_score
                consider(path + suffix_moves, complete_score, root_index=node.root_index)
                upper = next_score + color_ideal_upper_bound(next_board, self.rules)
                if upper <= best_score:
                    continue
                child = _SearchNode(
                    board=next_board,
                    score_so_far=next_score,
                    path=path,
                    lower_bound_total=complete_score,
                    upper_bound_total=upper,
                    completion=suffix_moves,
                    completion_score=suffix_score,
                    root_index=node.root_index,
                    topology_key=self._topology_key(next_board, node.root_index, self.config.min_group),
                )
                priority = (
                    -child.upper_bound_total,
                    -child.lower_bound_total,
                    -evaluate_board(next_board, min_group=self.config.min_group),
                )
                heapq.heappush(frontier_entries, (priority, frontier_counter, child))
                frontier_counter += 1

            if expanded % 32 == 0 and (rollout_budget is None or rollout_budget > 0):
                if rollout_budget is None or stats.complete_rollouts < rollout_budget:
                    suffix_started_at = stats.states_examined
                    random_completion = self._rollout(
                        node.board,
                        policy="random",
                        rng=rng,
                        budget=budget,
                        bounded=True,
                        stats=stats,
                    )
                    root_search_nodes[node.root_index] += stats.states_examined - suffix_started_at
                    if random_completion is not None:
                        consider(
                            node.path + random_completion[0],
                            node.score_so_far + random_completion[1],
                            root_index=node.root_index,
                        )
            if expanded % 16 == 0:
                emit()
            # Exhaustive mode is the only mode allowed to claim a proof. It
            # must retain every still-live branch; trimming here would turn a
            # bounded beam into an incomplete search while leaving the final
            # frontier empty-looking and making an invalid proof possible.
            if (
                self.config.quality.lower() != "exhaustive"
                and len(frontier_entries) > max(2 * self.config.value("beam_width"), 64)
            ):
                frontier_entries = self._trim_frontier(
                    frontier_entries,
                    limit=self.config.value("beam_width"),
                )

        stats.interrupted = stats.interrupted or budget.expired()
        proven = not root_moves or (
            self.config.quality.lower() == "exhaustive"
            and not stats.interrupted
            and not frontier_entries
            and len(root_evaluations) == len(root_moves)
        )
        if best_score < 0:
            # An empty board is already a complete terminal generation.
            best_score = 0
            best_moves = ()
        remaining_bounds = [item.upper_bound for item in root_evaluations.values()]
        remaining_bounds.extend(node.upper_bound_total for _, _, node in frontier_entries)
        upper_bound = max([best_score, global_upper, *remaining_bounds], default=best_score)
        stats.best_complete_score = best_score
        stats.upper_bound = best_score if proven else upper_bound
        solution = finish_solution(
            moves=best_moves,
            score=best_score,
            stats=stats,
            solver_name=f"hybrid-{self.config.quality.lower()}",
            optimal_proven=proven,
            upper_bound=best_score if proven else upper_bound,
        )
        updated_roots = tuple(
            replace(
                evaluation,
                best_complete_score_found=root_best.get(index, (evaluation.best_complete_score_found, evaluation.best_complete_plan))[0],
                best_complete_plan=root_best.get(index, (evaluation.best_complete_score_found, evaluation.best_complete_plan))[1],
                search_nodes=root_search_nodes.get(index, evaluation.search_nodes),
            )
            for index, evaluation in root_evaluations.items()
        )
        self.last_root_evaluations = tuple(
            sorted(updated_roots, key=lambda item: item.best_complete_score_found, reverse=True)
        )
        solution = replace(
            solution,
            root_evaluations=self.last_root_evaluations,
            solver_stats={
                **dict(solution.solver_stats),
                "quality": self.config.preset.name,
                "phases": [
                    "incumbents",
                    "root exploration",
                    "diverse rollouts",
                    "tree search",
                    "exact suffixes",
                    "rollouts",
                ],
                "root_moves": len(root_moves),
                "root_moves_evaluated": len(root_evaluations),
                "frontier_remaining": len(frontier_entries),
                "tree_nodes": expanded,
                "optimality_proven": proven,
                "upper_bound": upper_bound,
            },
        )
        emit(done=True)
        return solution
