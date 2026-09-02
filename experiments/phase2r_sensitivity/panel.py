"""Phase-2R 的面板构造：把存量 run 的 JSON 摊平成 (task, stage) 观测。

协议冻结于 038c238b22615e4fddbc4629feb9c8a9bcc9f70af7a5c0dcb5a6702843d530df，
在本文件存在之前。本模块不新增判据，只把 §2 的定义翻成计算。

**只读**：对 runs/ 一律以只读方式打开，不产生任何新的 runs/ 目录（§6）。

协议 §2 记录的陷阱：记录里的 `scope_size` 是该 scope 的**最终**规模
（对 False|True 恒为 4，即使在只有 WiC 一个成员的 stage 3），不是活跃规模。
活跃规模只能从 scope_radii[m]["members"] 取，本模块只走后者。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Obs:
    """一个 (seed, task, stage) 观测。字段名与协议 §2 逐一对应。"""

    seed: int
    task: str
    stage: int          # t，1..15
    pos: int            # pos(g)，g 被训练的位置
    age: int            # a = t - pos(g) >= 0
    scope: str
    m_live: int         # 该 stage 该 scope 的活跃成员数，取自 scope_radii
    is_singleton: bool  # 取自 run 自身的 singleton_scopes，不由标签串推断
    R_raw: float
    R_shr_global: float
    R_shr_scoped: float
    R_orc: float
    R_sso1: float | None

    @property
    def delta_id_global(self) -> float:
        return self.R_orc - self.R_shr_global

    @property
    def delta_id_scoped(self) -> float:
        return self.R_orc - self.R_shr_scoped

    @property
    def stale(self) -> float | None:
        """|R_sso1 − R_orc|。仅在单例 scope 上是纯陈旧度（Prop 1）。"""

        if self.R_sso1 is None:
            return None
        return abs(self.R_sso1 - self.R_orc)


def live_members(stage: dict, scope: str, m: str = "1") -> list[str]:
    """该 stage 该 scope 的活跃成员。唯一允许的 m 来源（协议 §2、§6）。"""

    entry = stage.get("scope_radii", {}).get(m, {}).get(scope)
    if entry is None:
        return []
    return list(entry.get("members", []))


def build_panel(path: Path, seed: int | None = None) -> list[Obs]:
    """把一个 seed 的 run JSON 摊成观测列表。只读打开。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload["stages"]
    singles = set(payload.get("singleton_scopes", []))

    # pos(g) 自 trained_task 反推，不硬编码任务顺序（§6）。
    pos = {s["trained_task"]: int(s["position"]) for s in stages}
    if seed is None:
        seed = int(payload.get("args", {}).get("seed", 0))

    out: list[Obs] = []
    for stage in stages:
        t = int(stage["position"])
        for name, info in stage["tasks"].items():
            if not info.get("scorable"):
                continue
            if name not in pos:
                continue
            scope = info["scope"]
            age = t - pos[name]
            if age < 0:                      # §6：age 必须非负
                raise ValueError(f"negative age for {name} at stage {t}")
            out.append(Obs(
                seed=seed, task=name, stage=t, pos=pos[name], age=age,
                scope=scope,
                m_live=len(live_members(stage, scope)),
                is_singleton=scope in singles,
                R_raw=float(info["R_raw"]),
                R_shr_global=float(info["R_shr_global"]),
                R_shr_scoped=float(info["R_shr_scoped"]),
                R_orc=float(info["R_orc"]),
                R_sso1=None if info.get("R_sso1") is None else float(info["R_sso1"]),
            ))
    return out
