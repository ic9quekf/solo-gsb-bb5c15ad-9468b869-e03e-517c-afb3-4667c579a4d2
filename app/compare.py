"""两轮评审会话对照：判断归一化、差异分类与含并列修正的 Kendall τ-b。

仅服务于「会话对照报告」，全部为纯计算，不读写任何会话数据：

- 每条判断先归一化为**按候选名称排序的无向组合 → 绝对胜者**
  （平局的绝对胜者为 None），因此两轮候选录入顺序不同也能逐组合对齐；
- 差异分四类：结果不变、胜者反转、胜负变平局、平局变胜负；
- 基于两轮最终 Schulze 排名的**全部候选对**计算 Kendall τ-b
  （含并列修正），分别统计同序 / 逆序 / 仅一侧并列数；
  修正分母为零（至少一侧全部并列）时 τ-b 返回 None。
"""
from __future__ import annotations

import math
from itertools import combinations

from .schulze import rank_groups, strongest_paths

# 差异类型
SAME = "same"                    # 结果不变（同向胜负，或双方均平局）
WINNER_REVERSED = "winner_reversed"  # 胜者反转（双方均为严格胜负，但胜者相反）
WIN_TO_TIE = "win_to_tie"        # 胜负变平局（A 有胜者，B 平局）
TIE_TO_WIN = "tie_to_win"        # 平局变胜负（A 平局，B 有胜者）

DIFF_TYPES = (SAME, WINNER_REVERSED, WIN_TO_TIE, TIE_TO_WIN)


def normalized_verdicts(history: list[dict]) -> dict[tuple[str, str], str | None]:
    """把判断记录归一化为「按名称排序的无向组合 → 绝对胜者」。

    存储记录 (a, b, result) 中的 result 相对存储顺序的 a 解释；这里先解出
    绝对胜者名称（平局为 None），再把组合按名称字典序排列。归一化结果与
    候选录入顺序无关，两轮会话可仅凭候选名称逐组合对齐。
    """
    verdicts: dict[tuple[str, str], str | None] = {}
    for h in history:
        a, b, result = h["a"], h["b"], h["result"]
        winner = None if result == "tie" else (a if result == "a" else b)
        verdicts[tuple(sorted((a, b)))] = winner
    return verdicts


def classify_pairs(
    pairs: list[tuple[str, str]],
    verdicts_a: dict[tuple[str, str], str | None],
    verdicts_b: dict[tuple[str, str], str | None],
) -> tuple[dict[str, int], list[dict]]:
    """对每个无向组合分类两轮判断的差异，返回计数与明细。

    明细按组合名称字典序排列；每项含两侧的绝对胜者（平局为 None）。
    方向约定：A = 当前会话，B = 另一会话。
    """
    counts = {t: 0 for t in DIFF_TYPES}
    details: list[dict] = []
    for key in pairs:
        winner_a = verdicts_a.get(key)
        winner_b = verdicts_b.get(key)
        if winner_a == winner_b:
            # 双方均为平局（None == None）或绝对胜者为同一候选
            kind = SAME
        elif winner_a is None:
            kind = TIE_TO_WIN
        elif winner_b is None:
            kind = WIN_TO_TIE
        else:
            kind = WINNER_REVERSED
        counts[kind] += 1
        details.append(
            {
                "a": key[0],
                "b": key[1],
                "type": kind,
                "winner_a": winner_a,
                "winner_b": winner_b,
            }
        )
    return counts, details


def kendall_tau_b(
    candidates: list[str], groups_a: list[list[str]], groups_b: list[list[str]]
) -> dict:
    """基于两轮最终排名（有序并列组）在全部候选对上计算含并列修正的 Kendall τ-b。

    每个候选对按两轮中的相对名次归入：
    - 同序 concordant：两侧均非并列且方向一致；
    - 逆序 discordant：两侧均非并列且方向相反；
    - 仅一侧并列 tied_one：恰有一侧把两者置于同一并列组；
    - 双侧并列 tied_both：两侧都把两者并列（不计入 τ-b 的任一修正项）。

    τ-b = (同序 − 逆序) / sqrt((n0 − n_x)(n0 − n_y))，其中 n_x / n_y 分别为
    A / B 轮中的并列对数（含双侧并列）。分母为 0（至少一侧全部候选并列）时
    τ-b 无定义，返回 null；同序与逆序也均为 0 时结果自然为 0.0。
    """
    rank_a = {name: gi for gi, group in enumerate(groups_a) for name in group}
    rank_b = {name: gi for gi, group in enumerate(groups_b) for name in group}

    concordant = discordant = tied_one = tied_both = 0
    tied_only_in_a = tied_only_in_b = 0
    for x, y in combinations(candidates, 2):
        ca = (rank_a[x] > rank_a[y]) - (rank_a[x] < rank_a[y])
        cb = (rank_b[x] > rank_b[y]) - (rank_b[x] < rank_b[y])
        if ca == 0 and cb == 0:
            tied_both += 1
        elif ca == 0:
            tied_only_in_a += 1
            tied_one += 1
        elif cb == 0:
            tied_only_in_b += 1
            tied_one += 1
        elif ca == cb:
            concordant += 1
        else:
            discordant += 1

    total = len(candidates) * (len(candidates) - 1) // 2
    # A 轮非并列对数 = 同序 + 逆序 + 仅 B 并列；B 轮同理
    not_tied_a = concordant + discordant + tied_only_in_b
    not_tied_b = concordant + discordant + tied_only_in_a
    denominator = math.sqrt(not_tied_a * not_tied_b)
    tau = None if denominator == 0 else (concordant - discordant) / denominator

    return {
        "tau": tau,
        "concordant": concordant,
        "discordant": discordant,
        "tied_one": tied_one,
        "tied_both": tied_both,
        "pairs": total,
    }


def compare_rounds(
    candidates_a: list[str],
    history_a: list[dict],
    candidates_b: list[str],
    history_b: list[dict],
) -> dict:
    """组装两轮已完成评审的完整对照报告（调用方须已校验候选集合相同且均完成）。

    报告包含：两侧最终 Schulze 排名、四类差异计数、全部组合的差异明细
    （按名称排序）以及基于最终排名的 Kendall τ-b。纯只读计算。
    """
    # 以名称排序作为组合枚举与明细展示的统一口径，与候选录入顺序无关
    names = sorted(set(candidates_a))
    pairs = list(combinations(names, 2))
    verdicts_a = normalized_verdicts(history_a)
    verdicts_b = normalized_verdicts(history_b)
    counts, details = classify_pairs(pairs, verdicts_a, verdicts_b)

    # 最终排名仍按各会话自己的候选录入序送入 Schulze（排名分组只取决于比较关系）
    groups_a = rank_groups(candidates_a, strongest_paths(candidates_a, history_a))
    groups_b = rank_groups(candidates_b, strongest_paths(candidates_b, history_b))
    tau = kendall_tau_b(names, groups_a, groups_b)

    return {
        "candidates": names,
        "ranking_a": groups_a,
        "ranking_b": groups_b,
        "summary": {"total": len(pairs), **counts},
        "tau_b": tau,
        "details": details,
    }
