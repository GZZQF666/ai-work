"""
甘蔗垂度预测模型
==============================
功能：
  1. 加载或生成甘蔗表型数据（叶片面积、茎粗、株高 → 预测垂度）
  2. 数据探索：相关性热图、特征分布图
  3. 多模型训练与评估（随机森林、梯度提升、SVR、线性回归、Ridge）
  4. 超参数调优（随机森林 GridSearchCV）
  5. 特征重要性分析
  6. 最优模型保存（joblib）
  7. 新样本垂度预测（命令行 / CSV 批量）
  8. 完整可视化报告图

输入特征：
  - leaf_area_cm2    叶片总面积（cm²）
  - stem_diameter_mm 茎（秆）直径（mm）
  - plant_height_cm  株高（cm）
  - leaf_count       叶片数量
  - leaf_length_cm   平均叶长（cm）
  - leaf_width_cm    平均叶宽（cm）

预测目标：
  - lodging_angle_deg  垂度（弯曲角度，°，0=直立，90=完全倒伏）

依赖：numpy pandas matplotlib seaborn scikit-learn joblib
安装：pip install numpy pandas matplotlib seaborn scikit-learn joblib
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════
FEATURES = [
    "leaf_area_cm2",
    "stem_diameter_mm",
    "plant_height_cm",
    "leaf_count",
    "leaf_length_cm",
    "leaf_width_cm",
]
TARGET = "lodging_angle_deg"
MODEL_PATH = "sugarcane_lodging_model.pkl"
FEATURE_LABEL = {
    "leaf_area_cm2":    "Leaf Area (cm2)",
    "stem_diameter_mm": "Stem Diameter (mm)",
    "plant_height_cm":  "Plant Height (cm)",
    "leaf_count":       "Leaf Count",
    "leaf_length_cm":   "Leaf Length (cm)",
    "leaf_width_cm":    "Leaf Width (cm)",
    "lodging_angle_deg": "Lodging Angle (deg)",
}

# ═══════════════════════════════════════════════════════
# 1. 数据生成 / 加载
# ═══════════════════════════════════════════════════════

def simulate_data(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    模拟甘蔗表型数据，使用农学先验关系生成垂度（弯曲角度）。
    
    垂度驱动逻辑（农学依据）：
      - 株高越高 → 重心上移 → 垂度增大
      - 茎粗越粗 → 抗弯能力强 → 垂度减小
      - 叶片面积越大 → 风阻增加 → 垂度增大
      - 叶片数越多 → 冠层重量增加 → 垂度增大
    """
    rng = np.random.default_rng(seed)

    leaf_area    = rng.normal(800,  200,  n_samples).clip(200,  1800)
    stem_diam    = rng.normal(30,   8,    n_samples).clip(10,   60)
    plant_height = rng.normal(250,  60,   n_samples).clip(80,   450)
    leaf_count   = rng.integers(6,  20,   n_samples).astype(float)
    leaf_length  = rng.normal(120,  30,   n_samples).clip(40,   220)
    leaf_width   = rng.normal(6,    1.5,  n_samples).clip(2,    12)

    # 垂度角度公式（含噪声）
    lodging = (
        0.06  * plant_height
        - 0.80 * stem_diam
        + 0.012 * leaf_area
        + 1.2  * leaf_count
        + 0.05 * leaf_length
        + 2.0  * leaf_width
        + rng.normal(0, 5, n_samples)
    ).clip(0, 90)

    df = pd.DataFrame({
        "leaf_area_cm2":    leaf_area,
        "stem_diameter_mm": stem_diam,
        "plant_height_cm":  plant_height,
        "leaf_count":       leaf_count,
        "leaf_length_cm":   leaf_length,
        "leaf_width_cm":    leaf_width,
        TARGET:             lodging,
    })
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    加载用户的 CSV/Excel 数据。
    
    期望列名（不区分大小写）：
      leaf_area_cm2, stem_diameter_mm, plant_height_cm,
      leaf_count, leaf_length_cm, leaf_width_cm, lodging_angle_deg
    
    若部分列缺失，缺失特征将以均值填充（用于预测场景）。
    """
    if filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # 列名标准化（忽略大小写和空格）
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  [提示] 以下列不存在，将用 0 填充（仅特征列）：{missing_cols}")
        for c in missing_cols:
            if c in FEATURES:
                df[c] = 0.0

    return df


# ═══════════════════════════════════════════════════════
# 2. 数据探索可视化
# ═══════════════════════════════════════════════════════

def plot_eda(df: pd.DataFrame, out_dir: str):
    """生成数据探索图：分布直方图 + 与垂度的散点图 + 相关热图"""
    os.makedirs(out_dir, exist_ok=True)

    # --- 图1：特征分布 ---
    cols = FEATURES + [TARGET]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=25, color="#378ADD", alpha=0.75, edgecolor="white")
        axes[i].set_title(FEATURE_LABEL.get(col, col), fontsize=11)
        axes[i].set_xlabel("Value", fontsize=9)
        axes[i].set_ylabel("Count", fontsize=9)
        axes[i].spines[["top", "right"]].set_visible(False)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 图2：各特征与垂度散点 ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors = ["#E24B4A", "#378ADD", "#1D9E75", "#BA7517", "#9F77DD", "#0F6E56"]
    for i, feat in enumerate(FEATURES):
        axes[i].scatter(df[feat], df[TARGET], alpha=0.45, s=20,
                        color=colors[i], edgecolors="none")
        # 趋势线
        z = np.polyfit(df[feat], df[TARGET], 1)
        p = np.poly1d(z)
        xs = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(xs, p(xs), color="#2C2C2A", linewidth=1.5, linestyle="--")
        axes[i].set_xlabel(FEATURE_LABEL.get(feat, feat), fontsize=10)
        axes[i].set_ylabel("Lodging Angle (deg)", fontsize=10)
        r = df[[feat, TARGET]].corr().iloc[0, 1]
        axes[i].set_title(f"r = {r:.3f}", fontsize=11)
        axes[i].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feature vs Lodging Angle (Scatter + Trend)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_scatter_vs_lodging.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 图3：相关热图 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    labels = [FEATURE_LABEL.get(c, c) for c in corr.columns]
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, xticklabels=labels, yticklabels=labels,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  EDA 图表已保存至 {out_dir}/")


# ═══════════════════════════════════════════════════════
# 3. 模型训练与评估
# ═══════════════════════════════════════════════════════

MODELS = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=4, random_state=42),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVR(kernel="rbf", C=10, epsilon=0.5)),
    ]),
}


def evaluate_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """训练所有模型并返回评估指标表"""
    results = []
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        results.append({
            "Model": name,
            "R2":    r2_score(y_test, y_pred),
            "MAE":   mean_absolute_error(y_test, y_pred),
            "RMSE":  np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV_R2": cv_r2,
        })
        trained[name] = (model, y_pred)
        print(f"  {name:<22} R²={r2_score(y_test, y_pred):.4f}  MAE={mean_absolute_error(y_test, y_pred):.2f}  RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    return pd.DataFrame(results).sort_values("R2", ascending=False), trained


def tune_best_model(X_train, y_train) -> RandomForestRegressor:
    """对最优模型（随机森林）进行超参数网格搜索"""
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [None, 8, 15],
        "min_samples_split": [2, 5],
        "max_features":  ["sqrt", "log2"],
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"\n  最优超参数：{gs.best_params_}")
    print(f"  CV R²（调优后）：{gs.best_score_:.4f}")
    return gs.best_estimator_


# ═══════════════════════════════════════════════════════
# 4. 可视化：评估结果
# ═══════════════════════════════════════════════════════

def plot_evaluation(results_df, trained_models, X_test, y_test, out_dir):
    """模型对比柱状图 + 最优模型预测 vs 实测散点图"""

    # --- 图4：模型对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["R2", "MAE", "RMSE"]
    bar_colors = ["#378ADD", "#E24B4A", "#1D9E75"]
    for ax, metric, c in zip(axes, metrics, bar_colors):
        df_plot = results_df.sort_values(metric, ascending=(metric != "R2"))
        bars = ax.barh(df_plot["Model"], df_plot[metric], color=c, alpha=0.8)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f"Model Comparison — {metric}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, df_plot[metric]):
            ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 图5：最优模型 预测 vs 实测 ---
    best_name = results_df.iloc[0]["Model"]
    best_model, y_pred_best = trained_models[best_name]
    r2 = r2_score(y_test, y_pred_best)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.6, s=40, color="#E24B4A", edgecolors="white",
               linewidths=0.4, label="Samples")
    lims = [min(y_test.min(), y_pred_best.min()) - 3,
            max(y_test.max(), y_pred_best.max()) + 3]
    ax.plot(lims, lims, color="#2C2C2A", linewidth=1.5, linestyle="--", label="1:1 line")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Measured Lodging Angle (deg)", fontsize=12)
    ax.set_ylabel("Predicted Lodging Angle (deg)", fontsize=12)
    ax.set_title(f"Best Model: {best_name}\nR² = {r2:.4f}  RMSE = {rmse:.2f}°", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/05_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  最优模型：{best_name}（R²={r2:.4f}，RMSE={rmse:.2f}°）")
    return best_model, best_name


def plot_feature_importance(model, feature_names, out_dir):
    """特征重要性条形图（仅适用于树模型）"""
    if not hasattr(model, "feature_importances_"):
        print("  [提示] 该模型不支持特征重要性，跳过")
        return

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance)))
    bars = ax.barh(importance.index, importance.values, color=colors)
    ax.set_xlabel("Feature Importance (Impurity-based)", fontsize=11)
    ax.set_title("Feature Importance for Lodging Prediction", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, importance.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_yticklabels([FEATURE_LABEL.get(f, f) for f in importance.index], fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  特征重要性图已保存")


def plot_residuals(model, X_test, y_test, out_dir):
    """残差分析图"""
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, color="#378ADD", edgecolors="none")
    axes[0].axhline(0, color="#E24B4A", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Value (deg)", fontsize=11)
    axes[0].set_ylabel("Residual (deg)", fontsize=11)
    axes[0].set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(residuals, bins=25, color="#1D9E75", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("Residual (deg)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  残差分析图已保存")


# ═══════════════════════════════════════════════════════
# 5. 新样本垂度预测
# ═══════════════════════════════════════════════════════

def predict_new(model, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    对新样本进行垂度预测。
    input_data: DataFrame，需包含 FEATURES 中的列
    """
    X = input_data[FEATURES].copy()
    preds = model.predict(X)
    result = input_data.copy()
    result["predicted_lodging_angle_deg"] = np.round(preds, 2)

    # 垂度等级
    def level(v):
        if v < 20:   return "Upright (直立)"
        elif v < 40: return "Slight (轻度倾斜)"
        elif v < 60: return "Moderate (中度垂度)"
        else:        return "Severe (严重垂倒)"

    result["lodging_level"] = result["predicted_lodging_angle_deg"].apply(level)
    return result


# ═══════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════

def main(
    train_data_path: str = None,
    predict_data_path: str = None,
    out_dir: str = "sugarcane_results",
    tune: bool = False,
):
    """
    主流程。

    参数
    ----
    train_data_path   训练数据 CSV/Excel 路径（含 lodging_angle_deg 列）
    predict_data_path 待预测新数据路径（可不含 lodging_angle_deg）
    out_dir           结果输出目录
    tune              是否对最优模型做超参数网格搜索调优
    """
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("  甘蔗垂度预测模型 — 训练与推理系统")
    print("=" * 60)

    # ── 1. 加载训练数据 ────────────────────────────────────
    print("\n[1/7] 加载训练数据...")
    if train_data_path and os.path.exists(train_data_path):
        df = load_data(train_data_path)
        print(f"  从文件加载：{train_data_path}，共 {len(df)} 条样本")
    else:
        print("  使用模拟数据（300 株甘蔗样本）")
        df = simulate_data(n_samples=300, seed=42)

    df.to_csv(f"{out_dir}/training_data.csv", index=False)
    print(f"  样本总数：{len(df)}")
    print(f"  垂度范围：{df[TARGET].min():.1f}° ~ {df[TARGET].max():.1f}°，"
          f"均值={df[TARGET].mean():.1f}°")

    # ── 2. 数据探索 ────────────────────────────────────────
    print("\n[2/7] 数据探索...")
    plot_eda(df, out_dir)
    print(f"\n  特征与垂度的 Pearson 相关系数：")
    corr = df[FEATURES].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)
    for feat, r in corr.items():
        print(f"  {FEATURE_LABEL.get(feat, feat):<25} r = {r:+.4f}")

    # ── 3. 划分训练 / 测试集 ───────────────────────────────
    print("\n[3/7] 划分数据集（8:2）...")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")

    # ── 4. 多模型训练评估 ──────────────────────────────────
    print("\n[4/7] 训练并评估多个模型...")
    results_df, trained_models = evaluate_models(X_train, X_test, y_train, y_test)
    results_df.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    print(f"\n  模型排名：")
    print(results_df.to_string(index=False))

    # ── 5. 超参数调优（可选）─────────────────────────────────
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name][0]

    if tune:
        print("\n[5/7] 超参数调优（随机森林 GridSearchCV，请稍候）...")
        tuned_rf = tune_best_model(X_train, y_train)
        y_pred_tuned = tuned_rf.predict(X_test)
        tuned_r2 = r2_score(y_test, y_pred_tuned)
        print(f"  调优后测试集 R²：{tuned_r2:.4f}")
        if tuned_r2 > results_df.iloc[0]["R2"]:
            best_model = tuned_rf
            best_model_name = "Random Forest (tuned)"
            print("  [更新] 使用调优后的随机森林作为最终模型")
    else:
        print("\n[5/7] 跳过超参数调优（使用 --tune 开启）")

    # ── 6. 结果可视化 ──────────────────────────────────────
    print("\n[6/7] 生成评估可视化图表...")
    final_model, final_name = plot_evaluation(results_df, trained_models,
                                               X_test, y_test, out_dir)
    if tune:
        final_model = best_model

    plot_feature_importance(
        final_model if hasattr(final_model, "feature_importances_")
        else trained_models.get("Random Forest", (None,))[0],
        FEATURES, out_dir
    )
    plot_residuals(final_model, X_test, y_test, out_dir)

    # ── 7. 保存模型 ────────────────────────────────────────
    model_save_path = f"{out_dir}/{MODEL_PATH}"
    joblib.dump(final_model, model_save_path)
    print(f"\n  最终模型已保存：{model_save_path}")

    # ── 8. 新样本预测 ─────────────────────────────────────
    print("\n[7/7] 新样本垂度预测...")
    if predict_data_path and os.path.exists(predict_data_path):
        new_df = load_data(predict_data_path)
        print(f"  从文件加载待预测数据：{predict_data_path}（{len(new_df)} 条）")
    else:
        # 示例新样本（手动输入 / 模拟场景演示）
        new_df = pd.DataFrame({
            "leaf_area_cm2":    [600, 900, 1200, 500, 1500],
            "stem_diameter_mm": [25,  35,  20,   40,  18],
            "plant_height_cm":  [200, 280, 320,  180, 380],
            "leaf_count":       [10,  14,  16,   8,   18],
            "leaf_length_cm":   [100, 130, 150,  90,  170],
            "leaf_width_cm":    [5,   6.5, 7,    4.5, 8],
        })
        print("  使用内置示例新样本（5 株）")

    pred_result = predict_new(final_model, new_df)
    pred_save_path = f"{out_dir}/prediction_results.csv"
    pred_result.to_csv(pred_save_path, index=False)
    print(f"\n  预测结果：")
    print(pred_result[["leaf_area_cm2", "stem_diameter_mm", "plant_height_cm",
                         "predicted_lodging_angle_deg", "lodging_level"]].to_string(index=False))
    print(f"\n  预测结果已保存：{pred_save_path}")

    # ── 完成 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  分析完成！所有结果保存至：{out_dir}/")
    print("=" * 60)
    print("""
  输出文件说明：
  ├── 01_feature_distributions.png — 特征分布图
  ├── 02_scatter_vs_lodging.png    — 特征 vs 垂度散点
  ├── 03_correlation_heatmap.png   — 相关热图
  ├── 04_model_comparison.png      — 模型性能对比
  ├── 05_pred_vs_actual.png        — 预测 vs 实测
  ├── 06_feature_importance.png    — 特征重要性
  ├── 07_residual_analysis.png     — 残差分析
  ├── model_comparison.csv         — 模型指标汇总
  ├── training_data.csv            — 训练数据（含模拟/原始）
  ├── prediction_results.csv       — 新样本预测结果
  └── sugarcane_lodging_model.pkl  — 最优模型文件
""")


# ═══════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="甘蔗垂度预测模型 — 机器学习回归",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用模拟数据训练并预测
  python sugarcane_lodging_model.py

  # 使用真实训练数据
  python sugarcane_lodging_model.py --train my_data.csv

  # 训练 + 对新数据预测
  python sugarcane_lodging_model.py --train train.csv --predict new_samples.csv

  # 开启超参数调优
  python sugarcane_lodging_model.py --train train.csv --tune

  # 指定输出目录
  python sugarcane_lodging_model.py --out results/

数据格式（CSV 列名）：
  leaf_area_cm2, stem_diameter_mm, plant_height_cm,
  leaf_count, leaf_length_cm, leaf_width_cm, lodging_angle_deg
""",
    )
    parser.add_argument("--train",   type=str, default=None,
                        help="训练数据路径（CSV/Excel），不提供则用模拟数据")
    parser.add_argument("--predict", type=str, default=None,
                        help="待预测新数据路径（CSV/Excel），不提供则用内置示例")
    parser.add_argument("--out",     type=str, default="sugarcane_results",
                        help="结果输出目录（默认 sugarcane_results）")
    parser.add_argument("--tune",    action="store_true",
                        help="开启随机森林超参数调优（较慢）")
    args = parser.parse_args()

    main(
        train_data_path=args.train,
        predict_data_path=args.predict,
        out_dir=args.out,
        tune=args.tune,
    )
