"""
转录组学 PCA（主成分分析）分析程序
=====================================
功能：
  1. 生成或加载模拟转录组基因表达矩阵
  2. 数据预处理：过滤低表达基因、归一化（CPM）、对数变换
  3. 特征缩放：StandardScaler
  4. PCA 降维分析
  5. 多图可视化：
     - PCA 2D 散点图（PC1 vs PC2）
     - PCA 3D 散点图
     - Scree Plot（碎石图）
     - 累计方差贡献图
     - 载荷图（Top 基因贡献）
     - 热图（Top 变异基因）
  6. 导出结果：坐标 CSV、主成分载荷 CSV、统计报告 TXT

依赖：numpy, pandas, matplotlib, seaborn, scikit-learn
安装：pip install numpy pandas matplotlib seaborn scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings
import os

warnings.filterwarnings("ignore")

# ── 中文字体设置（可选，若系统有中文字体则生效）────────────────────────────
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════════════
# 1. 模拟数据生成 / 数据加载
# ═══════════════════════════════════════════════════════════════════

def simulate_expression_data(
    n_samples: int = 60,
    n_genes: int = 2000,
    n_groups: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    模拟基因表达计数矩阵（行=基因，列=样本）。
    组间差异基因占 20%，模拟真实转录组 count 数据分布。
    """
    rng = np.random.default_rng(random_seed)
    group_labels = [f"Group{i + 1}" for i in range(n_groups)]
    samples_per_group = n_samples // n_groups

    sample_names = []
    group_info = []
    for g in group_labels:
        for i in range(samples_per_group):
            sample_names.append(f"{g}_S{i + 1:02d}")
            group_info.append(g)

    gene_names = [f"Gene_{i + 1:04d}" for i in range(n_genes)]

    # 基础表达量（负二项分布近似）
    counts = rng.negative_binomial(5, 0.3, size=(n_genes, len(sample_names))).astype(float)

    # 注入组间差异（前 20% 基因）
    diff_genes = int(n_genes * 0.20)
    for gi, g in enumerate(group_labels):
        idx = [j for j, s in enumerate(group_info) if s == g]
        fold = rng.uniform(3, 8, size=diff_genes) * (gi + 1)
        counts[:diff_genes, :][:, idx] *= fold[:, np.newaxis]

    df = pd.DataFrame(counts, index=gene_names, columns=sample_names)
    meta = pd.Series(group_info, index=sample_names, name="Group")
    return df, meta


def load_expression_data(filepath: str):
    """
    从 CSV/TSV 文件加载表达矩阵。
    期望格式：行=基因，列=样本，第一列为基因名称。
    metadata 需另存为 <filepath>_meta.csv，含 Sample 和 Group 两列。
    """
    sep = "\t" if filepath.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(filepath, index_col=0, sep=sep)

    meta_path = filepath.rsplit(".", 1)[0] + "_meta.csv"
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path, index_col=0)
        meta = meta_df["Group"]
    else:
        print(f"  [警告] 未找到 metadata 文件 {meta_path}，将所有样本标记为 Group1")
        meta = pd.Series("Group1", index=df.columns, name="Group")

    return df, meta


# ═══════════════════════════════════════════════════════════════════
# 2. 数据预处理
# ═══════════════════════════════════════════════════════════════════

def preprocess(
    counts: pd.DataFrame,
    min_count: int = 10,
    min_samples: int = 3,
    normalize: bool = True,
    log_transform: bool = True,
    pseudo_count: float = 1.0,
) -> pd.DataFrame:
    """
    预处理转录组 count 数据：
      1. 过滤低表达基因
      2. CPM 归一化（可选）
      3. log2 变换（可选）
    """
    print(f"  原始矩阵：{counts.shape[0]} 基因 × {counts.shape[1]} 样本")

    # 过滤：至少在 min_samples 个样本中 count >= min_count
    mask = (counts >= min_count).sum(axis=1) >= min_samples
    filtered = counts.loc[mask]
    print(f"  过滤后：{filtered.shape[0]} 基因（去除 {counts.shape[0] - filtered.shape[0]} 个低表达基因）")

    if normalize:
        # CPM（Counts Per Million）
        lib_size = filtered.sum(axis=0)
        cpm = filtered.div(lib_size, axis=1) * 1e6
    else:
        cpm = filtered.copy()

    if log_transform:
        expr = np.log2(cpm + pseudo_count)
    else:
        expr = cpm.copy()

    return expr


# ═══════════════════════════════════════════════════════════════════
# 3. 高变异基因选取
# ═══════════════════════════════════════════════════════════════════

def select_hvg(expr: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """
    选取方差最高的 top_n 个基因（High Variable Genes）用于 PCA。
    """
    gene_var = expr.var(axis=1)
    top_genes = gene_var.nlargest(top_n).index
    print(f"  选取 top {top_n} 高变异基因用于 PCA")
    return expr.loc[top_genes]


# ═══════════════════════════════════════════════════════════════════
# 4. PCA 分析
# ═══════════════════════════════════════════════════════════════════

def run_pca(expr_hvg: pd.DataFrame, n_components: int = 20):
    """
    执行 PCA：
      - 输入：高变异基因表达矩阵（基因 × 样本）
      - 标准化后进行 PCA
      - 返回：坐标 DataFrame、PCA 对象、标准化器
    """
    # PCA 要求 样本 × 基因
    X = expr_hvg.T  # shape: (n_samples, n_genes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, index=X.index, columns=pc_cols)

    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    print(f"\n  PCA 完成：{n_components} 个主成分")
    print(f"  PC1 方差贡献：{explained[0]:.2f}%")
    print(f"  PC2 方差贡献：{explained[1]:.2f}%")
    print(f"  PC3 方差贡献：{explained[2]:.2f}%")
    print(f"  前 5 PC 累计：{cumulative[4]:.2f}%")
    print(f"  前 10 PC 累计：{cumulative[9]:.2f}%")

    return pca_df, pca, scaler, explained, cumulative


# ═══════════════════════════════════════════════════════════════════
# 5. 可视化
# ═══════════════════════════════════════════════════════════════════

# 调色板（转录组学常用）
PALETTE = {
    "Group1": "#E24B4A",
    "Group2": "#378ADD",
    "Group3": "#1D9E75",
    "Group4": "#BA7517",
    "Group5": "#9F77DD",
}
DEFAULT_COLOR = "#888780"


def get_colors(groups: pd.Series) -> list:
    unique = groups.unique()
    color_map = {}
    colors_list = list(PALETTE.values())
    for i, g in enumerate(sorted(unique)):
        color_map[g] = colors_list[i % len(colors_list)]
    return [color_map[g] for g in groups], color_map


def plot_pca_2d(pca_df, meta, explained, save_path="pca_2d.png"):
    """PCA 二维散点图（PC1 vs PC2）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors, color_map = get_colors(meta)

    scatter = ax.scatter(
        pca_df["PC1"], pca_df["PC2"],
        c=colors, s=80, alpha=0.85, edgecolors="white", linewidths=0.6,
    )

    # 图例
    patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in sorted(color_map.items())
    ]
    ax.legend(handles=patches, title="Group", fontsize=10, title_fontsize=10,
              framealpha=0.8, edgecolor="#D3D1C7")

    ax.set_xlabel(f"PC1  ({explained[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2  ({explained[1]:.1f}%)", fontsize=12)
    ax.set_title("PCA — PC1 vs PC2", fontsize=14, fontweight="bold")
    ax.axhline(0, color="#D3D1C7", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#D3D1C7", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


def plot_pca_3d(pca_df, meta, explained, save_path="pca_3d.png"):
    """PCA 三维散点图（PC1/PC2/PC3）"""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors, color_map = get_colors(meta)

    ax.scatter(
        pca_df["PC1"], pca_df["PC2"], pca_df["PC3"],
        c=colors, s=70, alpha=0.85, edgecolors="white", linewidths=0.4,
    )

    patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in sorted(color_map.items())
    ]
    ax.legend(handles=patches, title="Group", fontsize=9)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=10, labelpad=8)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=10, labelpad=8)
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)", fontsize=10, labelpad=8)
    ax.set_title("PCA 3D — PC1/PC2/PC3", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


def plot_scree(explained, cumulative, save_path="pca_scree.png"):
    """碎石图 + 累计方差贡献图"""
    n = len(explained)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x, explained, color="#378ADD", alpha=0.75, label="各 PC 方差贡献")
    ax1.set_xlabel("主成分（Principal Component）", fontsize=12)
    ax1.set_ylabel("方差贡献率 (%)", fontsize=12, color="#378ADD")
    ax1.tick_params(axis="y", labelcolor="#378ADD")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x], rotation=45, fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#E24B4A", marker="o", markersize=5,
             linewidth=2, label="累计贡献率")
    ax2.axhline(80, color="#BA7517", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(n * 0.95, 81, "80%", color="#BA7517", fontsize=9, ha="right")
    ax2.set_ylabel("累计贡献率 (%)", fontsize=12, color="#E24B4A")
    ax2.tick_params(axis="y", labelcolor="#E24B4A")
    ax2.set_ylim(0, 105)

    ax1.set_title("Scree Plot — 主成分方差贡献", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
               fontsize=10, framealpha=0.85)

    ax1.spines[["top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


def plot_loadings(pca, gene_names, top_n=20, save_path="pca_loadings.png"):
    """
    PC1/PC2 载荷图：展示对主成分贡献最大的基因（双标图风格）
    """
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=gene_names,
        columns=["PC1_loading", "PC2_loading"],
    )
    # 按绝对值之和排序，选 top_n
    loadings["score"] = loadings["PC1_loading"].abs() + loadings["PC2_loading"].abs()
    top = loadings.nlargest(top_n, "score")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        loadings["PC1_loading"], loadings["PC2_loading"],
        color="#D3D1C7", s=15, alpha=0.5, zorder=1,
    )
    ax.scatter(
        top["PC1_loading"], top["PC2_loading"],
        color="#E24B4A", s=40, alpha=0.9, zorder=2, label="Top 基因",
    )
    for gene, row in top.iterrows():
        ax.annotate(
            gene,
            xy=(row["PC1_loading"], row["PC2_loading"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7.5, color="#3C3489",
        )

    ax.axhline(0, color="#B4B2A9", linewidth=0.8)
    ax.axvline(0, color="#B4B2A9", linewidth=0.8)
    ax.set_xlabel("PC1 loading", fontsize=12)
    ax.set_ylabel("PC2 loading", fontsize=12)
    ax.set_title(f"PCA Loadings — Top {top_n} 贡献基因（PC1/PC2）", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


def plot_heatmap(expr_hvg: pd.DataFrame, meta: pd.Series, top_n: int = 50,
                 save_path="pca_heatmap.png"):
    """Top 高变异基因热图（样本按分组着色）"""
    top_genes = expr_hvg.var(axis=1).nlargest(top_n).index
    data = expr_hvg.loc[top_genes]

    # 分组颜色条
    groups = sorted(meta.unique())
    group_palette = {g: v for g, v in zip(groups, list(PALETTE.values())[:len(groups)])}
    col_colors = meta.map(group_palette)

    g = sns.clustermap(
        data,
        col_colors=col_colors,
        cmap="RdYlBu_r",
        standard_scale=0,
        figsize=(12, 9),
        xticklabels=False,
        yticklabels=(top_n <= 60),
        linewidths=0,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.18),
    )
    g.ax_heatmap.set_xlabel("Samples", fontsize=11)
    g.ax_heatmap.set_ylabel("Genes", fontsize=11)
    g.fig.suptitle(f"Top {top_n} High Variable Genes Heatmap", fontsize=13,
                   fontweight="bold", y=1.01)

    # 手动图例
    patches = [mpatches.Patch(color=c, label=gr) for gr, c in sorted(group_palette.items())]
    g.ax_col_colors.legend(
        handles=patches, title="Group",
        loc="upper right", bbox_to_anchor=(1.25, 1.5),
        fontsize=9, title_fontsize=9,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


def plot_pc_pairs(pca_df, meta, explained, n_pc=4, save_path="pca_pairs.png"):
    """PC 配对散点矩阵（前 n_pc 个主成分两两散点）"""
    df = pca_df.iloc[:, :n_pc].copy()
    df["Group"] = meta.values

    colors, color_map = get_colors(meta)

    fig, axes = plt.subplots(n_pc, n_pc, figsize=(3 * n_pc, 3 * n_pc))
    fig.suptitle(f"PCA Pairs Plot — 前 {n_pc} 个主成分", fontsize=14, fontweight="bold")

    for i in range(n_pc):
        for j in range(n_pc):
            ax = axes[i][j]
            if i == j:
                # 对角线：直方图
                for g, c in color_map.items():
                    idx = df["Group"] == g
                    ax.hist(df.iloc[idx.values, i], bins=12, color=c, alpha=0.6)
                ax.set_ylabel(f"PC{i+1}\n({explained[i]:.1f}%)", fontsize=9)
            else:
                ax.scatter(
                    df.iloc[:, j], df.iloc[:, i],
                    c=colors, s=30, alpha=0.7, edgecolors="none",
                )
            ax.spines[["top", "right"]].set_visible(False)
            if i == n_pc - 1:
                ax.set_xlabel(f"PC{j+1}\n({explained[j]:.1f}%)", fontsize=9)

    patches = [mpatches.Patch(color=c, label=g) for g, c in sorted(color_map.items())]
    fig.legend(handles=patches, title="Group", loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存：{save_path}")


# ═══════════════════════════════════════════════════════════════════
# 6. 导出结果
# ═══════════════════════════════════════════════════════════════════

def export_results(
    pca_df: pd.DataFrame,
    pca: PCA,
    gene_names,
    meta: pd.Series,
    explained: np.ndarray,
    cumulative: np.ndarray,
    out_dir: str = "pca_results",
):
    os.makedirs(out_dir, exist_ok=True)

    # PCA 坐标
    coords_out = pca_df.copy()
    coords_out.insert(0, "Group", meta)
    coords_out.to_csv(f"{out_dir}/pca_coordinates.csv")
    print(f"  已保存：{out_dir}/pca_coordinates.csv")

    # 载荷矩阵
    n_pc = pca.n_components_
    pc_cols = [f"PC{i+1}" for i in range(n_pc)]
    loadings_df = pd.DataFrame(pca.components_.T, index=gene_names, columns=pc_cols)
    loadings_df.to_csv(f"{out_dir}/pca_loadings.csv")
    print(f"  已保存：{out_dir}/pca_loadings.csv")

    # 文本报告
    report_path = f"{out_dir}/pca_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  转录组学 PCA 分析报告\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"样本数：{pca_df.shape[0]}\n")
        f.write(f"分析主成分数：{n_pc}\n\n")
        f.write("主成分方差贡献率：\n")
        f.write(f"{'PC':<6}{'方差贡献 (%)':>14}{'累计 (%)':>12}\n")
        f.write("-" * 34 + "\n")
        for i, (ev, cum) in enumerate(zip(explained, cumulative)):
            f.write(f"PC{i+1:<4}{ev:>14.3f}{cum:>12.3f}\n")
        f.write("\n")

        # 达到 80% 需要多少 PC
        n80 = int(np.searchsorted(cumulative, 80.0)) + 1
        f.write(f"累计 80% 方差所需 PC 数：{n80}\n")
        f.write(f"累计 90% 方差所需 PC 数：{int(np.searchsorted(cumulative, 90.0)) + 1}\n\n")

        # 每组中心
        f.write("各组 PCA 中心坐标（均值）：\n")
        centers = pca_df.join(meta).groupby("Group")[["PC1", "PC2", "PC3"]].mean()
        f.write(centers.to_string() + "\n")

    print(f"  已保存：{report_path}")


# ═══════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════

def main(
    data_path: str = None,
    n_hvg: int = 500,
    n_pca_components: int = 20,
    out_dir: str = "pca_results",
):
    """
    主函数入口。

    参数
    ----
    data_path : str, optional
        基因表达矩阵 CSV/TSV 路径（行=基因，列=样本）。
        若为 None，则自动生成模拟数据。
    n_hvg : int
        用于 PCA 的高变异基因数量（默认 500）。
    n_pca_components : int
        计算的主成分数量（默认 20）。
    out_dir : str
        输出目录。
    """
    print("=" * 55)
    print("  转录组学 PCA 分析程序启动")
    print("=" * 55)

    # ── 1. 数据加载 ───────────────────────────────────────────────
    print("\n[1/6] 加载数据...")
    if data_path and os.path.exists(data_path):
        counts, meta = load_expression_data(data_path)
        print(f"  从文件加载：{data_path}")
    else:
        print("  使用模拟数据（60 样本 × 2000 基因，3 组）")
        counts, meta = simulate_expression_data(
            n_samples=60, n_genes=2000, n_groups=3, random_seed=42
        )

    print(f"  样本分组：{meta.value_counts().to_dict()}")

    # ── 2. 预处理 ─────────────────────────────────────────────────
    print("\n[2/6] 数据预处理...")
    expr = preprocess(counts, min_count=10, min_samples=3,
                      normalize=True, log_transform=True)

    # ── 3. 高变异基因 ─────────────────────────────────────────────
    print("\n[3/6] 选取高变异基因...")
    expr_hvg = select_hvg(expr, top_n=n_hvg)

    # ── 4. PCA ────────────────────────────────────────────────────
    print("\n[4/6] 执行 PCA 分析...")
    pca_df, pca, scaler, explained, cumulative = run_pca(expr_hvg, n_components=n_pca_components)

    # ── 5. 可视化 ─────────────────────────────────────────────────
    print("\n[5/6] 生成可视化图表...")
    os.makedirs(out_dir, exist_ok=True)

    plot_pca_2d(pca_df, meta, explained,
                save_path=f"{out_dir}/01_pca_2d.png")
    plot_pca_3d(pca_df, meta, explained,
                save_path=f"{out_dir}/02_pca_3d.png")
    plot_scree(explained, cumulative,
               save_path=f"{out_dir}/03_scree_plot.png")
    plot_loadings(pca, expr_hvg.index,
                  save_path=f"{out_dir}/04_pca_loadings.png")
    plot_heatmap(expr_hvg, meta, top_n=50,
                 save_path=f"{out_dir}/05_heatmap.png")
    plot_pc_pairs(pca_df, meta, explained, n_pc=4,
                  save_path=f"{out_dir}/06_pca_pairs.png")

    # ── 6. 导出 ───────────────────────────────────────────────────
    print("\n[6/6] 导出结果文件...")
    export_results(pca_df, pca, expr_hvg.index, meta, explained, cumulative,
                   out_dir=out_dir)

    print("\n" + "=" * 55)
    print(f"  分析完成！所有结果保存至：{out_dir}/")
    print("=" * 55)
    print("""
  输出文件说明：
  ├── 01_pca_2d.png        — PCA 二维散点图（PC1 vs PC2）
  ├── 02_pca_3d.png        — PCA 三维散点图（PC1/PC2/PC3）
  ├── 03_scree_plot.png    — 碎石图（方差贡献率）
  ├── 04_pca_loadings.png  — 基因载荷图
  ├── 05_heatmap.png       — 高变异基因热图
  ├── 06_pca_pairs.png     — PC 配对矩阵图
  ├── pca_coordinates.csv  — 样本 PCA 坐标
  ├── pca_loadings.csv     — 基因主成分载荷
  └── pca_report.txt       — 统计报告
""")


# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="转录组学 PCA 分析程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="基因表达矩阵文件路径（CSV/TSV）。若不提供则使用模拟数据。",
    )
    parser.add_argument(
        "--hvg", type=int, default=500,
        help="用于 PCA 的高变异基因数量（默认 500）",
    )
    parser.add_argument(
        "--n_pc", type=int, default=20,
        help="计算的主成分数量（默认 20）",
    )
    parser.add_argument(
        "--out", type=str, default="pca_results",
        help="输出目录（默认 pca_results）",
    )
    args = parser.parse_args()

    main(
        data_path=args.data,
        n_hvg=args.hvg,
        n_pca_components=args.n_pc,
        out_dir=args.out,
    )
