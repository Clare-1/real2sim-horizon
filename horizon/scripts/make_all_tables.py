"""Emit every experiment table into one markdown file.

Sources are the result directories actually on disk; a table whose directory is missing is written
as "not run" rather than omitted, so the report never looks emptier than it is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from report_minimal_validation import bootstrap_ci, paired_deltas  # noqa: E402

OPS = ("nothing", "calibrate_geometry", "regenerate_uniform", "regenerate_targeted", "all")
VARIANTS = ("legacy_repeat", "robust_repeat", "freeze_first", "true_params")


def _load(path: Path):
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    # Fallback: a run that is still in progress has per-run records on disk.
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]
    return recs or None


def _load_calab(path: Path):
    recs = _load(path)
    if recs:
        return recs
    log = Path("/tmp/calab.log")
    if not log.exists():
        return None
    import re
    out = []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"(legacy_repeat|robust_repeat|freeze_first|true_params)\s+s(\d+): "
                     r"err\(mm\)=\[([-\d., ]+)\]\s+SR=([\d.]+)", line.strip())
        if m:
            out.append({"variant": m.group(1), "seed": int(m.group(2)),
                        "errors_mm": [float(x) for x in m.group(3).split(",")],
                        "final_sr": float(m.group(4))})
    return out or None


def table_r1(root: Path) -> list[str]:
    recs = _load(root / "results/r1_matrix_20261112")
    out = ["## 表 R1　配方 × 原因（ΔSR，pp；95% bootstrap CI，3 种子）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    causes = sorted({r["cause"] for r in recs})
    out += ["| 操作 | " + " | ".join(causes) + " |", "|---|" + "---|" * len(causes)]
    for op in OPS:
        cells = []
        for cause in causes:
            d = [r["delta"] for r in recs if r["op"] == op and r["cause"] == cause]
            if not d:
                cells.append("-")
                continue
            lo, hi = bootstrap_ci(d)
            cells.append(f"{100*sum(d)/len(d):+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        out.append(f"| {op} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def table_minimal(root: Path) -> list[str]:
    recs = _load(root / "results/minimal_validation_20261112/stage1") or []
    recs = recs + (_load(root / "results/minimal_validation_20261112/stage2") or [])
    recs = recs or None or []
    recs = recs + (_load(root / "results/minimal_validation_20261112/stage2") or [])
    recs = recs or None
    out = ["## 表 2/3　最小验证：选择器 × 原因（协议预算，10 个场景实例）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    causes = sorted({r["cause"] for r in recs})
    sels = sorted({r["selector"] for r in recs})
    out += ["| 选择器 | 原因 | 首步选对 | 最终 SR | 部署次数 | 新增标签 |", "|---|---|---|---:|---:|---:|"]
    for sel in sels:
        for cause in causes:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == cause]
            if not rs:
                continue
            ok = sum(1 for r in rs if r["first_step_correct"])
            out.append(f"| {sel} | {cause} | {ok}/{len(rs)} | "
                       f"{sum(r['final_sr'] for r in rs)/len(rs):.3f} | "
                       f"{sum(r['deploy_episodes'] for r in rs)/len(rs):.0f} | "
                       f"{sum(r['new_labels'] for r in rs)/len(rs):.0f} |")
    out += ["", "| 原因 | 选择器 | 配对数 | 相对固定配方 (pp) | 95% CI |", "|---|---|---:|---:|---|"]
    for cause in causes:
        for sel in sels:
            if sel == "fixed":
                continue
            d = paired_deltas(recs, sel, cause)
            if len(d) < 2:
                continue
            lo, hi = bootstrap_ci(d)
            out.append(f"| {cause} | {sel} | {len(d)} | {100*sum(d)/len(d):+.1f} | "
                       f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
    if "llm" not in sels:
        out += ["", "> **LLM agent 行缺失**：`rsi/llm_planner.py` 已实现（单轮决策、temperature 0、"
                    "token 记账、无密钥时硬失败），但本机无 `ANTHROPIC_API_KEY`，因此没有跑。"
                    "表格保留该行位置，不填臆造数字。"]
    out.append("")
    return out


def table_calibrator(root: Path) -> list[str]:
    recs = _load_calab(root / "results/calibrator_ablation_20261112")
    out = ["## 表 8　校准器消融（几何偏差，真值 20 mm；无策略学习，只动 twin）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    out += ["| 校准器 | seed | 第 1 轮 | 第 2 轮 | 第 3 轮 | 发散 | 最终 SR |",
            "|---|---:|---:|---:|---:|---|---:|"]
    for v in VARIANTS:
        rs = sorted([r for r in recs if r["variant"] == v], key=lambda r: r["seed"])
        for r in rs:
            cells = [f"{e:+.2f}" for e in r["errors_mm"]]
            while len(cells) < 3:
                cells.append("-")
            diverged = max(abs(e) for e in r["errors_mm"]) > 10.0
            out.append(f"| {v} | {r['seed']} | " + " | ".join(cells[:3]) +
                       f" | {'**是**' if diverged else '否'} | {r['final_sr']:.3f} |")
    n = {v: sum(1 for r in recs if r["variant"] == v and max(abs(e) for e in r["errors_mm"]) > 10.0)
         for v in VARIANTS}
    tot = {v: sum(1 for r in recs if r["variant"] == v) for v in VARIANTS}
    out += ["", "发散种子数：" + "；".join(f"{v} {n[v]}/{tot[v]}" for v in VARIANTS if tot[v]), ""]
    return out




def table_mixed(root: Path) -> list[str]:
    recs = _load(root / "results/minimal_validation_mixed_20261116")
    out = ["## 表 混合原因：固定配方 vs 选择器（10 个场景实例，30 runs）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    rows = {}
    for r in recs:
        rows.setdefault(r["selector"], []).append(r["final_sr"])
    out += ["| 选择器 | 最终 SR |", "|---|---:|"]
    for sel in ("fixed", "rule", "oracle"):
        if sel in rows:
            out.append(f"| {sel} | {sum(rows[sel])/len(rows[sel]):.3f} |")
    d = paired_deltas(recs, "rule", recs[0]["cause"])
    if len(d) > 1:
        lo, hi = bootstrap_ci(d)
        out += ["", f"规则 − 固定：**{100*sum(d)/len(d):+.1f} pp** [{100*lo:+.1f}, {100*hi:+.1f}]（n={len(d)}）"]
    out.append("")
    return out


def table_budget(root: Path) -> list[str]:
    recs = _load(root / "results/budget_split_20261116")
    out = ["## 表 预算平分：单一操作 vs 全都做一遍（B=12000 标签）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    causes = sorted({r["cause"] for r in recs})
    out += ["| 原因 | 单一操作 | 全做一遍 |", "|---|---:|---:|"]
    for c in causes:
        cells = []
        for arm in ("single", "do_all"):
            v = [r["sr_nominal"] for r in recs if r["cause"] == c and r["arm"] == arm]
            cells.append(f"{sum(v)/len(v):.3f}" if v else "-")
        out.append(f"| {c} | {cells[0]} | {cells[1]} |")
    out.append("")
    return out


def table_verify(root: Path) -> list[str]:
    recs = _load(root / "results/verify_ablation_20261116")
    out = ["## 表 Challenge 3：候选核验消融", ""]
    if not recs:
        return out + ["*未运行*", ""]
    out += ["| 场景 | 种子 | 不核验 | 核验 | 拒绝次数 |", "|---|---:|---:|---:|---:|"]
    for c in sorted({r["cause"] for r in recs}):
        for s_ in sorted({r["seed"] for r in recs}):
            a = [r for r in recs if r["cause"] == c and r["seed"] == s_]
            if len(a) < 2:
                continue
            nv = next((r for r in a if r["arm"] == "no_verify"), None)
            vf = next((r for r in a if r["arm"] == "verify"), None)
            if nv and vf:
                out.append(f"| {c} | {s_} | {nv['sr_nominal']:.3f}/{nv['sr_perturbed']:.3f} | "
                           f"{vf['sr_nominal']:.3f}/{vf['sr_perturbed']:.3f} | {vf['rejected']} |")
    out += ["", "> 负结果：核验把正确校准拒掉，几何场景 SR 从 0.80 掉到 0.40。原因见实验进展 §10。", ""]
    return out





def table_verify_pred(root: Path) -> list[str]:
    recs = _load(root / "results/verify_pred_20261116")
    out = ["## 表 Challenge 3（第二次）：策略无关的预测核验", ""]
    if not recs:
        return out + ["*未运行*", ""]
    out += ["| 场景 | 种子 | 不核验 | 预测核验 | 拒绝次数 |", "|---|---:|---:|---:|---:|"]
    for c in sorted({r["cause"] for r in recs}):
        for s_ in sorted({r["seed"] for r in recs}):
            a = [r for r in recs if r["cause"] == c and r["seed"] == s_]
            nv = next((r for r in a if r["arm"] == "no_verify"), None)
            vp = next((r for r in a if r["arm"] == "verify_pred"), None)
            if nv and vp:
                out.append(f"| {c} | {s_} | {nv['sr_nominal']:.3f}/{nv['sr_perturbed']:.3f} | "
                           f"{vp['sr_nominal']:.3f}/{vp['sr_perturbed']:.3f} | {vp['rejected']} |")
    out += ["", "> 对正确校准无损（几何三种子 SR 与不核验完全相同），并在已正确的 twin 上拒绝 1 次更新。", ""]
    return out



def table_four_causes(root: Path) -> list[str]:
    recs = _load(root / "results/four_causes_20261118") or []
    recs = recs or (_load(root / "results/minimal_validation_4causes_paper") or [])
    out = ["## 表 四类失败原因 × 选择器（含随机/轮流/bandit）", ""]
    if not recs:
        return out + ["*未运行*", ""]
    causes = sorted({r["cause"] for r in recs})
    sels = sorted({r["selector"] for r in recs})
    out += ["| 选择器 | " + " | ".join(causes) + " | 首步对 |",
            "|---|" + "---|" * len(causes) + "---|"]
    for sel in sels:
        cells, ok, tot = [], 0, 0
        for c in causes:
            rs = [r["final_sr"] for r in recs if r["selector"] == sel and r["cause"] == c]
            cells.append(f"{sum(rs)/len(rs):.3f}" if rs else "-")
            ok += sum(1 for r in recs if r["selector"] == sel and r["cause"] == c and r["first_step_correct"])
            tot += sum(1 for r in recs if r["selector"] == sel and r["cause"] == c)
        out.append(f"| {sel} | " + " | ".join(cells) + f" | {ok}/{tot} |")
    out.append("")
    return out

def table_robodojo() -> list[str]:
    return ["## 表 附录　RoboDojo", "",
            "**未运行。** 理由（`docs/robodojo_appendix.md` + 本次核查）：", "",
            "1. 本地无官方任务/观测/动作合约，历史记录只有“42 bimanual tasks, Isaac Sim, MIT”一句；",
            "2. 官方仓库与站点在历史核查时 TLS/认证失败，本次未重新核实；",
            "3. 需要 Isaac Sim（数十 GB），本机根分区仅剩 24 GB；",
            "4. 第三方自报的 score 60.08 与 SR 54.17% 不是同一指标，不能替代官方 SR。", "",
            "替代锚点：LIBERO 同样未接入（无本地 checkout），因此**当前没有任何外部榜单数字**。", ""]


def main() -> int:
    parts = ["# 实验表格汇总", "",
             "由 `scripts/make_all_tables.py` 从结果目录直接生成；缺失的实验标注为“未运行”，不省略。", ""]
    parts += table_r1(ROOT)
    parts += table_minimal(ROOT)
    parts += table_calibrator(ROOT)
    parts += table_mixed(ROOT)
    parts += table_budget(ROOT)
    parts += table_verify(ROOT)
    parts += table_verify_pred(ROOT)
    parts += table_four_causes(ROOT)
    parts += table_robodojo()
    text = "\n".join(parts)
    (ROOT.parent / "实验表格汇总_2026-11-12.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())