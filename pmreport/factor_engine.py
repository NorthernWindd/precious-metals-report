from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .indicators import add_indicators


FEATURE_NAMES = (
    "RET",
    "LOG_VOL",
    "PRESSURE",
    "FOMO",
    "DEV",
    "VOL_CLUSTER",
    "MOM_REV",
    "REL_STRENGTH",
    "HL_RANGE",
    "CLOSE_POS",
    "VOL_TREND",
    "MACD_HIST",
    "BOLL_POS",
    "ATR_PCT",
)


def _clip(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(arr, lower, upper)


def _shift(arr: np.ndarray, periods: int) -> np.ndarray:
    if periods == 0:
        return arr
    if periods < 0:
        periods = -periods
        out = np.empty_like(arr)
        out[:, :-periods] = arr[:, periods:]
        out[:, -periods:] = 0.0
        return out
    out = np.empty_like(arr)
    out[:, periods:] = arr[:, :-periods]
    out[:, :periods] = 0.0
    return out


def _ts_delay(x: np.ndarray, d: int) -> np.ndarray:
    return _shift(x, d)


def _op_gate(condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(condition > 0, x, y)


def _op_jump(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    z = (x - mean) / std
    return np.maximum(z - 3.0, 0.0)


def _op_decay(x: np.ndarray) -> np.ndarray:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


OPS_CONFIG: list[tuple[str, Callable[..., np.ndarray], int]] = [
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", lambda x, y: x / (y + 1e-6), 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", np.abs, 1),
    ("SIGN", np.sign, 1),
    ("GATE", _op_gate, 3),
    ("JUMP", _op_jump, 1),
    ("DECAY", _op_decay, 1),
    ("DELAY1", lambda x: _ts_delay(x, 1), 1),
    ("MAX3", lambda x: np.maximum(x, np.maximum(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
]

OP_NAMES = tuple(name for name, _, _ in OPS_CONFIG)
FEATURE_COUNT = len(FEATURE_NAMES)


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    enriched = add_indicators(df)
    close = enriched["close"].astype(float)
    open_ = enriched.get("open", close)
    high = enriched.get("high", close)
    low = enriched.get("low", close)
    volume = enriched.get("volume", pd.Series(0.0, index=close.index)).astype(float)

    ret = np.log(close / close.shift(1).replace(0, np.nan))
    volume_prev = volume.shift(1).replace(0, np.nan)
    fomo_raw = (volume / volume_prev) - 1.0
    fomo = fomo_raw - fomo_raw.shift(1)

    ret_5 = close.pct_change(5)
    mom_prev = ret_5.shift(1)
    mom_rev = ((ret_5 * mom_prev) < 0).astype(float)

    vol_cluster = (ret**2).rolling(10, min_periods=2).mean().pow(0.5)
    rsi = enriched["rsi14"]
    rel_strength = (rsi - 50.0) / 50.0
    macd_hist = enriched["macd_hist"] / close.replace(0, np.nan)
    boll_pos = (close - enriched["boll_mid"]) / (
        enriched["boll_upper"] - enriched["boll_lower"]
    ).replace(0, np.nan)

    features = pd.DataFrame(index=enriched.index)
    features["RET"] = ret
    features["LOG_VOL"] = np.log1p(volume)
    features["PRESSURE"] = np.tanh(
        (close - open_) / (high - low).replace(0, np.nan) * 3.0
    )
    features["FOMO"] = _clip(fomo.to_numpy(dtype=float), -5.0, 5.0)
    features["DEV"] = close / close.rolling(20, min_periods=2).mean() - 1.0
    features["VOL_CLUSTER"] = vol_cluster
    features["MOM_REV"] = mom_rev
    features["REL_STRENGTH"] = rel_strength
    features["HL_RANGE"] = (high - low) / close.replace(0, np.nan)
    features["CLOSE_POS"] = (close - low) / (high - low).replace(0, np.nan)
    features["VOL_TREND"] = _clip(fomo_raw.to_numpy(dtype=float), -5.0, 5.0)
    features["MACD_HIST"] = macd_hist
    features["BOLL_POS"] = boll_pos
    features["ATR_PCT"] = enriched["atr_pct"]
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_feature_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, Any]:
    valid_frames = {
        symbol: bars_by_symbol.get(symbol)
        for symbol in symbols
        if symbol in bars_by_symbol and bars_by_symbol[symbol] is not None and not bars_by_symbol[symbol].empty
    }
    if not valid_frames:
        raise ValueError("没有可用于因子生成的行情数据")

    common_index = None
    for frame in valid_frames.values():
        index = pd.DatetimeIndex(frame.index)
        common_index = index if common_index is None else common_index.intersection(index)
    if common_index is None or len(common_index) < 60:
        raise ValueError("公共行情日期不足，无法进行因子生成")

    ordered_symbols = [symbol for symbol in symbols if symbol in valid_frames]
    layers: list[np.ndarray] = []
    for symbol in ordered_symbols:
        feature_frame = compute_feature_frame(valid_frames[symbol])
        feature_frame = feature_frame.reindex(common_index).fillna(0.0)
        layers.append(feature_frame[list(FEATURE_NAMES)].to_numpy(dtype=float).T)

    feature_tensor = np.stack(layers, axis=0)
    target_frame = pd.DataFrame(
        {
            symbol: valid_frames[symbol]["close"].reindex(common_index)
            for symbol in ordered_symbols
        }
    )
    close_panel = target_frame.to_numpy(dtype=float).T
    target_ret_1 = close_panel[:, 1:] / np.maximum(close_panel[:, :-1], 1e-9) - 1.0
    target_ret_1 = np.concatenate([target_ret_1, np.zeros((len(ordered_symbols), 1))], axis=1)
    target_ret_5 = np.empty_like(close_panel)
    target_ret_5[:, :-5] = close_panel[:, 5:] / np.maximum(close_panel[:, :-5], 1e-9) - 1.0
    target_ret_5[:, -5:] = 0.0

    return {
        "feature_tensor": feature_tensor,
        "symbols": ordered_symbols,
        "dates": common_index,
        "target_ret_1": target_ret_1,
        "target_ret_5": target_ret_5,
    }


def execute_formula(
    tokens: list[int],
    feature_tensor: np.ndarray,
) -> np.ndarray | None:
    stack: list[np.ndarray] = []
    try:
        for token in tokens:
            token = int(token)
            if token < FEATURE_COUNT:
                if token >= feature_tensor.shape[1]:
                    return None
                stack.append(feature_tensor[:, token, :])
                continue

            op_index = token - FEATURE_COUNT
            if op_index < 0 or op_index >= len(OPS_CONFIG):
                return None
            _, func, arity = OPS_CONFIG[op_index]
            if len(stack) < arity:
                return None
            args = [stack.pop() for _ in range(arity)]
            args.reverse()
            result = func(*args)
            result = np.nan_to_num(
                result,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            stack.append(result)
    except Exception:
        return None

    return stack[0] if len(stack) == 1 else None


def _rank_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        order = np.argsort(np.argsort(values))
        valid = np.isfinite(values)
        out[:, column] = order.astype(float)
        out[~valid, column] = np.nan
    return out


def _spearman_ic(signal: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    signal_rank = _rank_columns(signal)
    target_rank = _rank_columns(target)
    ics = []
    for column in range(signal.shape[1]):
        x = signal_rank[:, column]
        y = target_rank[:, column]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 4:
            continue
        if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
            ics.append(0.0)
            continue
        ics.append(float(np.corrcoef(x[mask], y[mask])[0, 1]))
    if not ics:
        return 0.0, 0.0
    arr = np.array(ics)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def _long_short_return(signal: np.ndarray, target: np.ndarray, quantile: float = 0.3) -> float:
    long_returns: list[float] = []
    short_returns: list[float] = []
    for column in range(signal.shape[1]):
        values = signal[:, column]
        returns = target[:, column]
        mask = np.isfinite(values) & np.isfinite(returns)
        if mask.sum() < 5:
            continue
        order = np.argsort(np.argsort(values[mask]))
        n = len(order)
        long_mask = order >= int(n * (1.0 - quantile))
        short_mask = order < int(n * quantile)
        long_returns.append(float(np.mean(returns[mask][long_mask])))
        short_returns.append(float(np.mean(returns[mask][short_mask])))
    if not long_returns or not short_returns:
        return 0.0
    return float(np.mean(long_returns) - np.mean(short_returns))


def evaluate_formula(
    tokens: list[int],
    panel: dict[str, Any],
) -> dict[str, Any] | None:
    signal = execute_formula(tokens, panel["feature_tensor"])
    if signal is None or not np.isfinite(signal).all():
        return None
    if float(np.nanstd(signal)) < 1e-5:
        return None

    ic1_mean, ic1_std = _spearman_ic(signal, panel["target_ret_1"])
    ic5_mean, ic5_std = _spearman_ic(signal, panel["target_ret_5"])
    ls1 = _long_short_return(signal, panel["target_ret_1"])
    ls5 = _long_short_return(signal, panel["target_ret_5"])
    fitness = (
        100.0 * abs(ic1_mean)
        + 100.0 * abs(ic5_mean)
        + 2.0 * max(0.0, ls1 * 100.0)
        + 2.0 * max(0.0, ls5 * 100.0)
    )
    return {
        "tokens": [int(token) for token in tokens],
        "ic1_mean": float(ic1_mean),
        "ic1_std": float(ic1_std),
        "ic5_mean": float(ic5_mean),
        "ic5_std": float(ic5_std),
        "long_short_1": float(ls1),
        "long_short_5": float(ls5),
        "fitness": float(fitness),
    }


@dataclass
class FormulaTree:
    op: int | None = None
    token: int | None = None
    children: list["FormulaTree"] | None = None

    def tokens(self) -> list[int]:
        if self.children:
            out: list[int] = []
            for child in self.children:
                out.extend(child.tokens())
            if self.op is None:
                raise ValueError("operator node missing op")
            out.append(self.op)
            return out
        if self.token is None:
            raise ValueError("leaf node missing token")
        return [self.token]


def _random_tree(rng: np.random.Generator, depth: int) -> FormulaTree:
    if depth <= 0 or rng.random() < 0.55:
        return FormulaTree(token=int(rng.integers(0, FEATURE_COUNT)))
    op_index = int(rng.integers(0, len(OPS_CONFIG)))
    _, _, arity = OPS_CONFIG[op_index]
    return FormulaTree(
        op=FEATURE_COUNT + op_index,
        children=[_random_tree(rng, depth - 1) for _ in range(arity)],
    )


def _mutate_tree(rng: np.random.Generator, tree: FormulaTree, depth: int) -> FormulaTree:
    if rng.random() < 0.45 or tree.children is None:
        return _random_tree(rng, depth)
    child_index = int(rng.integers(0, len(tree.children)))
    new_children = list(tree.children)
    new_children[child_index] = _mutate_tree(rng, new_children[child_index], depth - 1)
    return FormulaTree(op=tree.op, children=new_children)


def _crossover(
    rng: np.random.Generator,
    left: FormulaTree,
    right: FormulaTree,
) -> FormulaTree:
    if left.op is None and left.token is None and left.children is None:
        return left
    if right.op is None and right.token is None and right.children is None:
        return right

    left_copy = FormulaTree(op=left.op, token=left.token, children=list(left.children) if left.children else None)
    right_copy = FormulaTree(op=right.op, token=right.token, children=list(right.children) if right.children else None)

    def choose_subtree(node: FormulaTree) -> FormulaTree:
        if node.children is None or rng.random() < 0.4:
            return node
        return choose_subtree(node.children[int(rng.integers(0, len(node.children)))])

    return choose_subtree(right_copy) if rng.random() < 0.5 else left_copy


def mine_factors(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    population_size: int = 120,
    generations: int = 8,
    max_depth: int = 4,
    top_n: int = 10,
    seed: int = 7,
) -> list[dict[str, Any]]:
    panel = build_feature_panel(bars_by_symbol, symbols)
    rng = np.random.default_rng(seed)

    evaluated: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_tree(tree: FormulaTree) -> dict[str, Any] | None:
        tokens = tree.tokens()
        key = tuple(tokens)
        if key in evaluated:
            return evaluated[key]
        result = evaluate_formula(tokens, panel)
        evaluated[key] = result if result else {}
        return result

    population = [_random_tree(rng, max_depth) for _ in range(population_size)]
    scored: list[dict[str, Any]] = []
    for tree in population:
        result = evaluate_tree(tree)
        if result:
            scored.append(result)

    for _ in range(generations):
        scored.sort(key=lambda item: item["fitness"], reverse=True)
        elites = scored[: max(4, population_size // 4)]
        next_population: list[FormulaTree] = []
        for result in elites:
            next_population.append(_tree_from_tokens(result["tokens"], rng))
        while len(next_population) < population_size:
            parent = _tree_from_tokens(rng.choice(elites)["tokens"], rng)
            if rng.random() < 0.5:
                child = _mutate_tree(rng, parent, max_depth)
            else:
                other = _tree_from_tokens(rng.choice(elites)["tokens"], rng)
                child = _crossover(rng, parent, other)
            next_population.append(child)

        for tree in next_population:
            result = evaluate_tree(tree)
            if result and result not in scored:
                scored.append(result)

    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for result in scored:
        unique[tuple(result["tokens"])] = result
    ranked = sorted(unique.values(), key=lambda item: item["fitness"], reverse=True)
    return ranked[:top_n]


def _tree_from_tokens(tokens: list[int], rng: np.random.Generator) -> FormulaTree:
    # This function is only used for retaining already-valid elite formulas.
    # Rebuilding a tree from postfix tokens is not necessary for scoring,
    # so we wrap the formula in a simple holder that serializes back exactly.
    return _TokenHolder(tokens)


class _TokenHolder(FormulaTree):
    def __init__(self, tokens: list[int]):
        self._tokens = list(tokens)
        super().__init__()

    def tokens(self) -> list[int]:
        return list(self._tokens)


def compute_latest_factor_scores(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    factors: list[dict[str, Any]],
) -> dict[str, float]:
    if not factors:
        return {}
    panel = build_feature_panel(bars_by_symbol, symbols)
    factor_values = np.zeros((len(panel["symbols"]), len(factors)), dtype=float)
    for index, factor in enumerate(factors):
        signal = execute_formula(factor["tokens"], panel["feature_tensor"])
        if signal is None:
            factor_values[:, index] = np.nan
        else:
            factor_values[:, index] = signal[:, -1]

    ranks = np.zeros_like(factor_values)
    for column in range(factor_values.shape[1]):
        values = factor_values[:, column]
        ranks[:, column] = np.argsort(np.argsort(values)) / max(len(values) - 1, 1) * 100.0
    score = np.nanmean(ranks, axis=1)
    return {
        symbol: float(score[index])
        for index, symbol in enumerate(panel["symbols"])
        if np.isfinite(score[index])
    }


def load_factors(path: str | Path) -> list[dict[str, Any]]:
    factor_path = Path(path)
    if not factor_path.exists():
        return []
    data = json.loads(factor_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("factors", [])
    return data


def save_factors(path: str | Path, factors: list[dict[str, Any]]) -> None:
    factor_path = Path(path)
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    factor_path.write_text(
        json.dumps({"factors": factors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
