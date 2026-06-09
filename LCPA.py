#!/usr/bin/env python3
"""
LCPA - Local CV-improvement Path Analysis
==========================================

制約付き多目的最適化アルゴリズムの探索履歴から、CV (Constraint Violation) の
改善構造を解析し、実行可能領域への到達を阻害するボトルネック領域を特定するツール。

解析の流れ:
  1. Data_graph フォルダから指定問題の探索履歴 CSV を読み込む。
  2. DENSITY_PERCENTILE > 0 の場合、決定変数空間で局所密度が低い孤立点を除去する。
     (サンプリング不足による偽の局所最小 CV 点を除外するため。)
  3. 残った各個体について、自身より小さい CV を持つ個体のうち決定変数空間で
     最も近い個体へエッジを張り、CV 改善グラフ (CV-improvement graph) を構築する。
  4. 各ノードについて CV 改善先へのエッジ長 (決定変数空間距離) を計算する。
  5. 縦軸 CV / 横軸 Edge Length の decision graph を作成する。

低 CV 領域で CV 改善先までの距離が大きい個体 = CV 改善のボトルネック候補。
(候補は decision graph から目視で読み取る。自動検出は行わない。)

パラメータは下の「設定」ブロックを直接編集する。実行:
    python LCPA.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ファイル出力用 (GUI 不要)
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ========================================================================== #
# 設定
# ========================================================================== #
PROBLEM = "RWMOP22"          # 解析する問題名 ({DATA_DIR}/{PROBLEM}_solutions.csv)
DATA_DIR = "Data_graph"      # 探索履歴フォルダ
OUT_DIR = "LCPA_out"         # 出力フォルダ

# --- データ準備 ---
GEN_RANGE = None             # 利用世代範囲 (lo, hi) 両端含む。None で全世代
MAX_POINTS = 30000           # 利用最大個体数 (超過分はランダムサンプリング)
SCALER = "standard"          # 決定変数のスケーリング: "standard" / "minmax" / "none"
NORMALIZE_CV = False          # 各制約を正規化してから CV を合算するか
SEED = 0                     # サンプリング乱数シード

# --- 低密度点除去 ---
DENSITY_PERCENTILE = 5.0     # 除去する最低密度点の割合 [%]。0 で無効 (全点を使用)
DENSITY_K = 10               # 局所密度推定に用いる近傍数

# --- グラフ / プロット ---
INIT_K = 16                  # 条件付き最近傍探索の初期 k
LOGX = False                 # 横軸 (Edge Length) を対数スケールに
LOGY = False                  # 縦軸 (CV) を対数スケールに
# ========================================================================== #


# --------------------------------------------------------------------------- #
# データ構造
# --------------------------------------------------------------------------- #
@dataclass
class History:
    """読み込んだ探索履歴と派生情報をまとめたコンテナ。"""

    df: pd.DataFrame          # 元データ (サブサンプル後)
    x_cols: list[str]         # 決定変数列名
    f_cols: list[str]         # 目的関数列名
    con_cols: list[str]       # 制約列名
    cv: np.ndarray            # 各個体の CV
    X: np.ndarray             # 決定変数行列 (生値)
    Xn: np.ndarray            # 決定変数行列 (スケーリング後 = 距離計算に使用)


# --------------------------------------------------------------------------- #
# 1. データ読み込み
# --------------------------------------------------------------------------- #
def load_history() -> History:
    """設定された問題の探索履歴を読み込み、CV と決定変数行列を準備する。"""
    path = os.path.join(DATA_DIR, f"{PROBLEM}_solutions.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"探索履歴が見つかりません: {path}\n"
            f"  {DATA_DIR}/ の中身を確認してください。"
        )

    print(f"[load] reading {path}")
    df = pd.read_csv(path)
    print(f"[load] raw rows = {len(df):,}, columns = {list(df.columns)}")

    # 列を役割ごとに分類
    x_cols = [c for c in df.columns if c.lower().startswith("x_")]
    f_cols = [c for c in df.columns if c.lower().startswith("f_")]
    con_cols = [c for c in df.columns if c.lower().startswith("con")]

    if not x_cols:
        raise ValueError("決定変数列 (X_*) が見つかりません。")
    if not con_cols:
        raise ValueError("制約列 (Con_*) が見つかりません。CV を計算できません。")
    print(f"[load] decision vars = {x_cols}")
    print(f"[load] objectives   = {f_cols}")
    print(f"[load] constraints  = {con_cols}")

    # 世代フィルタ
    if GEN_RANGE is not None and "Gen" in df.columns:
        lo, hi = GEN_RANGE
        df = df[(df["Gen"] >= lo) & (df["Gen"] <= hi)].reset_index(drop=True)
        print(f"[load] gen filter [{lo},{hi}] -> {len(df):,} rows")

    # 欠損・無限値の除去
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=x_cols + con_cols
    ).reset_index(drop=True)

    # サブサンプリング
    if MAX_POINTS is not None and len(df) > MAX_POINTS:
        df = df.sample(n=MAX_POINTS, random_state=SEED).reset_index(drop=True)
        print(f"[load] subsampled -> {len(df):,} rows (seed={SEED})")

    # --- CV (Constraint Violation) の計算 ---
    # 制約は g(x) <= 0 形式。正の値が違反量。CV = sum(max(0, g_i)).
    con = df[con_cols].to_numpy(dtype=float)
    viol = np.clip(con, 0.0, None)
    if NORMALIZE_CV:
        # 各制約の違反量を最大違反で正規化してからスケールを揃えて合算
        col_max = viol.max(axis=0)
        col_max[col_max == 0] = 1.0
        viol = viol / col_max
    cv = viol.sum(axis=1)
    n_feasible = int((cv <= 0).sum())
    print(
        f"[load] CV: min={cv.min():.4g} max={cv.max():.4g} "
        f"mean={cv.mean():.4g} | feasible (CV<=0) = {n_feasible:,}"
    )

    # --- 決定変数空間のスケーリング (距離計算用) ---
    X = df[x_cols].to_numpy(dtype=float)
    if SCALER == "standard":
        Xn = StandardScaler().fit_transform(X)
    elif SCALER == "minmax":
        Xn = MinMaxScaler().fit_transform(X)
    elif SCALER == "none":
        Xn = X.copy()
    else:
        raise ValueError(f"未知の SCALER: {SCALER}")

    return History(
        df=df, x_cols=x_cols, f_cols=f_cols, con_cols=con_cols,
        cv=cv, X=X, Xn=Xn,
    )


# --------------------------------------------------------------------------- #
# 2. 低密度点除去
# --------------------------------------------------------------------------- #
def filter_low_density(Xn: np.ndarray) -> np.ndarray:
    """局所密度が低い孤立点を除去し、残す点のマスクを返す。

    DENSITY_PERCENTILE = 0 の場合は全点を残す。
    サンプリング不足で孤立した点が偽の局所最小 CV 点として検出されるのを防ぐ。

    Returns
    -------
    keep : np.ndarray (bool)
        残す点のマスク (True = 残す)。
    """
    n = len(Xn)
    if not DENSITY_PERCENTILE or DENSITY_PERCENTILE <= 0:
        print(f"[filter] DENSITY_PERCENTILE=0 -> using all {n:,} individuals")
        return np.ones(n, dtype=bool)

    kth = _kth_nn_distance(Xn, k=DENSITY_K)
    # 距離が大きい = 低密度。上位 DENSITY_PERCENTILE% を除去。
    thr = np.percentile(kth, 100.0 - DENSITY_PERCENTILE)
    keep = kth <= thr
    print(f"[filter] low-density removal (sparsest {DENSITY_PERCENTILE}%): "
          f"{n:,} -> {int(keep.sum()):,}")
    return keep


def _kth_nn_distance(Xn: np.ndarray, k: int = 10) -> np.ndarray:
    """各点の k 番目の最近傍距離を返す (自身を除く)。"""
    tree = cKDTree(Xn)
    kk = min(k + 1, len(Xn))  # 自身を含めて k+1 個取得し、最後の列を使う
    dists, _ = tree.query(Xn, k=kk)
    if dists.ndim == 1:
        dists = dists[:, None]
    return dists[:, -1]


# --------------------------------------------------------------------------- #
# 4-5. CV 改善グラフの構築 (条件付き最近傍 / density-peaks の delta)
# --------------------------------------------------------------------------- #
def build_cv_improvement_graph(
    Xn: np.ndarray,
    cv: np.ndarray,
    keep: np.ndarray,
) -> pd.DataFrame:
    """各個体について「自身より CV が小さい個体のうち最近傍」へエッジを張る。

    効率化のため KDTree の k-NN を段階的に拡張して条件付き最近傍を探索する
    (全点が近傍に改善先を持つ通常ケースで高速)。改善先が k-NN 内に
    見つからない点のみ、残り候補に対して厳密に最近傍を求める。

    Returns
    -------
    graph : pd.DataFrame
        columns = [node, target, edge_length, cv, target_cv, is_root, orig_index]
        node/target は keep=True の個体に振り直したローカル添字。
        is_root=True は自身より小さい CV を持つ個体が存在しない個体 (グラフの根)。
    """
    idx = np.where(keep)[0]
    P = Xn[idx]
    C = cv[idx]
    n = len(idx)
    print(f"[graph] building CV-improvement graph over {n:,} individuals")

    if n == 0:
        return pd.DataFrame(
            columns=["node", "target", "edge_length", "cv", "target_cv",
                     "is_root", "orig_index"]
        )

    tree = cKDTree(P)
    target = np.full(n, -1, dtype=np.int64)
    edge_len = np.full(n, np.nan, dtype=float)

    unresolved = np.arange(n)
    k = min(INIT_K, n)
    while len(unresolved) > 0 and k <= n:
        dists, nbrs = tree.query(P[unresolved], k=k)
        if k == 1:  # 念のため (1 次元化対策)
            dists = dists[:, None]
            nbrs = nbrs[:, None]

        cv_unres = C[unresolved][:, None]
        cv_nbrs = C[nbrs]
        # 自身 (距離 0) を除外しつつ、CV が厳密に小さい近傍を改善先候補とする
        is_better = cv_nbrs < cv_unres  # (m, k)

        still = []
        for row, gi in enumerate(unresolved):
            better_cols = np.where(is_better[row])[0]
            if better_cols.size > 0:
                # k-NN は距離昇順なので最初の改善先が最近傍
                col = better_cols[0]
                target[gi] = nbrs[row, col]
                edge_len[gi] = dists[row, col]
            else:
                still.append(gi)

        unresolved = np.array(still, dtype=np.int64)
        if len(unresolved) == 0 or k == n:
            break
        k = min(k * 2, n)

    # k-NN 拡張で解決しなかった点 = 真に改善先が存在しない (CV 最小付近の) 根ノード、
    # あるいは数値的に全近傍を見ても無い点。厳密に確認する。
    is_root = np.zeros(n, dtype=bool)
    if len(unresolved) > 0:
        print(f"[graph] resolving {len(unresolved)} candidate root(s) exactly")
        for gi in unresolved:
            mask = C < C[gi]
            if not mask.any():
                is_root[gi] = True
                continue
            cand = np.where(mask)[0]
            d = np.linalg.norm(P[cand] - P[gi], axis=1)
            j = cand[np.argmin(d)]
            target[gi] = j
            edge_len[gi] = float(d.min())

    # 根ノードのエッジ長は (density-peaks 慣習に従い) 最大エッジ長に設定して可視化する
    if is_root.any():
        max_len = np.nanmax(edge_len[~is_root]) if (~is_root).any() else 0.0
        edge_len[is_root] = max_len
        print(f"[graph] root node(s) (global-min CV): {int(is_root.sum())} "
              f"-> edge_length set to max ({max_len:.4g})")

    # 入次数 (in-degree): そのノードを CV 改善先として指している個体数。
    # 大きいほど多くの個体の改善経路が合流する = 探索が集まりやすいノード。
    in_degree = np.bincount(target[target >= 0], minlength=n).astype(np.int64)

    graph = pd.DataFrame({
        "node": np.arange(n),
        "target": target,
        "edge_length": edge_len,
        "cv": C,
        "target_cv": np.where(target >= 0, C[np.clip(target, 0, n - 1)], np.nan),
        "in_degree": in_degree,
        "is_root": is_root,
        "orig_index": idx,  # 元 df 上の行番号
    })
    print(f"[graph] edges: {int((target >= 0).sum()):,}, "
          f"edge_length: min={np.nanmin(edge_len):.4g} "
          f"max={np.nanmax(edge_len):.4g} mean={np.nanmean(edge_len):.4g}")
    print(f"[graph] in-degree: max={in_degree.max()} mean={in_degree.mean():.2f} "
          f"| nodes with in-degree>0: {int((in_degree > 0).sum()):,}")
    return graph


# --------------------------------------------------------------------------- #
# 6. decision graph の描画
# --------------------------------------------------------------------------- #
def plot_decision_graph(graph: pd.DataFrame, out_path: str) -> None:
    """縦軸 CV / 横軸 Edge Length の decision graph を描画して保存。

    各ノードは入次数 (in-degree = そのノードを CV 改善先として指している個体数)
    で色付けする。入次数が大きいノード = 多くの個体の CV 改善経路が合流する点で、
    探索がそこに集まりやすいことを示す。根ノード (自身より小さい CV を持つ個体が
    無い = 実行可能 / 大域最小 CV の個体) は背景として淡色表示する。
    低 CV かつ Edge Length が大きい個体が CV 改善のボトルネック候補 (目視で判断)。
    """
    roots = graph[graph["is_root"]]
    nodes = graph[~graph["is_root"]]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 根ノード (主に実行可能個体) は淡い灰色で背景表示
    if len(roots) > 0:
        ax.scatter(roots["edge_length"], roots["cv"],
                   c="0.85", s=16, alpha=0.4, edgecolors="none",
                   label=f"root nodes (no smaller CV, n={len(roots)})")

    # CV 改善先を持つ個体を入次数で色付け (歪んだ分布のため log1p で正規化)
    if len(nodes) > 0:
        deg = nodes["in_degree"].to_numpy()
        order = np.argsort(deg)  # 入次数の大きいノードを前面に描画
        sc = ax.scatter(
            nodes["edge_length"].to_numpy()[order],
            nodes["cv"].to_numpy()[order],
            c=np.log1p(deg[order]), cmap="plasma",
            s=34, alpha=0.85, edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("in-degree (incoming CV-improvement edges)")
        # 色は log1p スケールだが目盛は実際の整数入次数で表示する
        dmax = int(deg.max())
        tick_degrees = sorted(set([0, 1, 2, 3, 5, 10, 20, 50, dmax]))
        tick_degrees = [d for d in tick_degrees if d <= dmax]
        cbar.set_ticks(np.log1p(tick_degrees))
        cbar.set_ticklabels([str(d) for d in tick_degrees])

    ax.set_xlabel("Edge Length (decision-space distance to CV-improvement target)")
    ax.set_ylabel("CV")
    ax.set_title(f"Decision Graph — {PROBLEM}  (color = in-degree)")
    if LOGX:
        ax.set_xscale("log")
    if LOGY:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved decision graph -> {out_path}")


def plot_histograms(graph: pd.DataFrame, out_path: str) -> None:
    """エッジ長と入次数のヒストグラムを描画して保存。

    - Edge Length 分布: 改善先までの距離の分布。右裾の長さがボトルネックの広がり。
    - In-degree 分布: 各ノードに集まる改善経路数の分布。裾の重いノード =
      多くの個体が合流する CV 改善のハブ。
    """
    nodes = graph[~graph["is_root"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- エッジ長ヒストグラム ---
    el = nodes["edge_length"].to_numpy()
    el = el[np.isfinite(el)]
    axes[0].hist(el, bins=50, color="tab:blue", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Edge Length (distance to CV-improvement target)")
    axes[0].set_ylabel("count (number of nodes)")
    axes[0].set_title("Edge Length distribution")
    axes[0].set_yscale("log")  # 大半が短く稀に長い (重い裾) ため対数
    axes[0].grid(True, alpha=0.3)

    # --- 入次数ヒストグラム (0 を含む全ノード) ---
    # 入次数は「指される側」の量なので根ノードも含めて集計する
    deg_all = graph["in_degree"].to_numpy()
    dmax = int(deg_all.max())
    bins = np.arange(0, dmax + 2) - 0.5  # 整数ビン
    axes[1].hist(deg_all, bins=bins, color="tab:orange", alpha=0.8,
                 edgecolor="white")
    axes[1].set_xlabel("In-degree (incoming CV-improvement edges)")
    axes[1].set_ylabel("count (number of nodes)")
    axes[1].set_title("In-degree distribution")
    axes[1].set_yscale("log")  # 0 が多数 & 裾が重いため対数
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{PROBLEM} — Edge Length & In-degree histograms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved histograms -> {out_path}")


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    hist = load_history()
    keep = filter_low_density(hist.Xn)
    graph = build_cv_improvement_graph(hist.Xn, hist.cv, keep)

    # 元データの情報を付与して保存
    enriched = graph.copy()
    oi = enriched["orig_index"].to_numpy()
    enriched["Gen"] = hist.df["Gen"].to_numpy()[oi] if "Gen" in hist.df else np.nan
    for c in hist.f_cols:
        enriched[c] = hist.df[c].to_numpy()[oi]

    graph_csv = os.path.join(OUT_DIR, f"{PROBLEM}_cv_graph.csv")
    enriched.to_csv(graph_csv, index=False)
    print(f"[save] CV-improvement graph -> {graph_csv}")

    plot_path = os.path.join(OUT_DIR, f"{PROBLEM}_decision_graph.png")
    plot_decision_graph(graph, plot_path)

    hist_path = os.path.join(OUT_DIR, f"{PROBLEM}_histograms.png")
    plot_histograms(graph, hist_path)


if __name__ == "__main__":
    run()
