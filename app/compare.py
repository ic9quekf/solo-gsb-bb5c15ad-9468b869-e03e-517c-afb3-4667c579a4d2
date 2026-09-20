"""会话对照报告：比较两个**已完成**评审会话的逐条判断与最终排名。

对照前提（由 API 层校验）：

- 两会话的**启用候选名称集合完全相同**（候选录入顺序可以不同）；
- 两轮评审均已完成（启用候选之间的每个组合都已有判断）。

归一化口径：每条判断按**候选名称字符串排序**为无向组合 (a, b)
（字典序 a 在 b 前），并把结果换算为**绝对胜者**名称（平局为
None）——与候选录入顺序、存储时的 a/b 方向均无关，因此两个会话
即使候选录入顺序不同，也能按名称逐组合对齐。

差异分四类（方向均为「本轮 A → 另一轮 B」）：

- ``same``：结果不变（同一绝对胜者，或两轮同为平局）；
- ``winner_reversed``：胜者反转（两轮都有胜者且恰好相反）；
- ``win_to_tie``：胜负变平局（本轮有胜者，另一轮为平局）；
- ``tie_to_win``：平局变胜负（本轮为平局，另一轮有胜者）。

排名一致性用**含并列修正的 Kendall τ-b** 度量：

    τ-b = (n_同序 − n_逆序) / sqrt((n_同序 + n_逆序 + n_仅A并列)
                                   (n_同序 + n_逆序 + n_仅B并列))

两侧都并列的候选对不计入分子分母；分母为 0（如某一侧全部候选
并列）时 ``value`` 返回 None（JSON null）。
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from .schulze import rank_groups, strongest_paths

SAME = "same"
WINNER_REVERSED = "winner_reversed"
WIN_TO_TIE = "win_to_tie"
TIE_TO_WIN = "tie_to_win"

# 三类「差异」（结果不变以外的全部类型）
DIFF_TYPES = (WINNER_REVERSED, WIN_TO_TIE, TIE_TO_WIN)
ALL_TYPES = (SAME, *DIFF_TYPES)


def normalized_verdicts(history: list[dict]) -> dict[tuple[str, str], str | None]:
    """把判断记录归一化为「按名称排序的无向组合 → 绝对胜者」。

    键为按名称字符串字典序排序的 (a, b)（a < b）；值为绝对胜者
    名称，平局为 None。同组合在库中只有一条记录（接口层已保证），
    这里对防御性重复以最后一条为准。
    """
    verdicts: dict[tuple[str, str], str | None] = {}
    for h in history:
        x, y = h["a"], h["b"]
        lo, hi = (x, y) if x <= y else (y, x)
        if h["result"] == "tie":
            winner: str | None = None
        else:
            # 存储 result 相对存储顺序的 a；换算为绝对胜者名称
            winner = x if h["result"] == "a" else y
        verdicts[(lo, hi)] = winner
    return verdicts


def is_complete(
    candidates: list[str], verdicts: dict[tuple[str, str], str | None]
) -> bool:
    """启用候选之间的每个名称序组合是否都已有判断。"""
    return all(
        pair in verdicts for pair in combinations(sorted(candidates), 2)
    )


def _diff_type(winner_a: str | None, winner_b: str | None) -> str:
    """两轮归一化结果的差异分类（方向：本轮 A → 另一轮 B）。"""
    if winner_a == winner_b:  # 同一胜者，或二者均为 None（同为平局）
        return SAME
    if winner_a is None:
        return TIE_TO_WIN
    if winner_b is None:
        return WIN_TO_TIE
    return WINNER_REVERSED


def _relative_result(lo: str, hi: str, winner: str | None) -> str:
    """绝对胜者换算回相对名称序组合 (lo, hi) 的结果：a/b/tie。"""
    if winner is None:
        return "tie"
    return "a" if winner == lo else "b"


def final_ranking(candidates: list[str], history: list[dict]) -> list[list[str]]:
    """一轮评审的 Schulze 最终排名（有序并列组）。"""
    return rank_groups(candidates, strongest_paths(candidates, history))


def kendall_tau_b(
    names: list[str],
    groups_a: list[list[str]],
    groups_b: list[list[str]],
) -> dict[str, Any]:
    """基于两轮最终排名的全部候选对计算含并列修正的 Kendall τ-b。

    分别统计：同序（concordant）、逆序（discordant）、仅本轮并列
    （tied_in_a_only）、仅另一轮并列（tied_in_b_only）、两侧均并列
    （tied_both）。对外把两种「仅一侧并列」合并为 tied_one_side，
    分母仍按标准 τ-b 公式分别取用两者。分母为 0 时 value 为 None。
    """
    rank_a = {
        name: group_index
        for group_index, group in enumerate(groups_a)
        for name in group
    }
    rank_b = {
        name: group_index
        for group_index, group in enumerate(groups_b)
        for name in group
    }

    concordant = discordant = 0
    tied_a_only = tied_b_only = tied_both = 0
    for x, y in combinations(names, 2):
        tied_in_a = rank_a[x] == rank_a[y]
        tied_in_b = rank_b[x] == rank_b[y]
        if tied_in_a and tied_in_b:
            tied_both += 1
        elif tied_in_a:
            tied_a_only += 1
        elif tied_in_b:
            tied_b_only += 1
        elif (rank_a[x] < rank_a[y]) == (rank_b[x] < rank_b[y]):
            concordant += 1
        else:
            discordant += 1

    # 标准 τ-b 分母的平方：(P+Q+T_A)(P+Q+T_B)；两侧均并列的对不参与
    factor_a = concordant + discordant + tied_a_only
    factor_b = concordant + discordant + tied_b_only
    value: float | None
    if factor_a <= 0 or factor_b <= 0:
        value = None  # 分母为 0（至少一侧排名无任何可比较的次序）
    else:
        value = (concordant - discordant) / math.sqrt(factor_a * factor_b)

    return {
        "concordant": concordant,
        "discordant": discordant,
        # 「仅一侧并列」合计计数（页面汇总展示用）
        "tied_one_side": tied_a_only + tied_b_only,
        # 分母需要的分项，一并返回便于核对
        "tied_in_a_only": tied_a_only,
        "tied_in_b_only": tied_b_only,
        "tied_both": tied_both,
        "total_pairs": concordant + discordant + tied_a_only + tied_b_only + tied_both,
        "value": value,
    }


def build_comparison(
    sid_a: str,
    candidates_a: list[str],
    history_a: list[dict],
    sid_b: str,
    candidates_b: list[str],
    history_b: list[dict],
) -> dict[str, Any]:
    """组装完整对照报告（前提：启用候选集合一致且两轮均已完成）。

    明细按**名称字典序**的无向组合排列，与任一会话的候选录入顺序
    无关；报告为纯只读计算，不改动任何会话数据。
    """
    names = sorted(candidates_a)  # 调用方已保证 set(candidates_a) == set(candidates_b)
    verdicts_a = normalized_verdicts(history_a)
    verdicts_b = normalized_verdicts(history_b)

    counts = {t: 0 for t in ALL_TYPES}
    details: list[dict[str, Any]] = []
    for lo, hi in combinations(names, 2):
        winner_a = verdicts_a[(lo, hi)]
        winner_b = verdicts_b[(lo, hi)]
        diff_type = _diff_type(winner_a, winner_b)
        counts[diff_type] += 1
        details.append(
            {
                # 组合统一按名称排序，result 相对名称序的 a（lo）解释
                "a": lo,
                "b": hi,
                "winner_a": winner_a,
                "winner_b": winner_b,
                "result_a": _relative_result(lo, hi, winner_a),
                "result_b": _relative_result(lo, hi, winner_b),
                "type": diff_type,
            }
        )

    groups_a = final_ranking(candidates_a, history_a)
    groups_b = final_ranking(candidates_b, history_b)

    return {
        "session_a": {
            "id": sid_a,
            "candidates": candidates_a,
            "ranking": groups_a,
        },
        "session_b": {
            "id": sid_b,
            "candidates": candidates_b,
            "ranking": groups_b,
        },
        # 归一化后的统一候选序列（名称字典序）
        "candidates": names,
        "summary": {
            "total_pairs": len(details),
            "same": counts[SAME],
            "winner_reversed": counts[WINNER_REVERSED],
            "win_to_tie": counts[WIN_TO_TIE],
            "tie_to_win": counts[TIE_TO_WIN],
            "different": sum(counts[t] for t in DIFF_TYPES),
        },
        "kendall_tau_b": kendall_tau_b(names, groups_a, groups_b),
        "details": details,
    }
