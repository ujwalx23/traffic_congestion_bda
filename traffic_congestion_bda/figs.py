"""
figs.py
Smart City Traffic Congestion Prediction Using Big Data
Presentation & Visual Analytics Generator

This script reads results.json and predictions_test.csv to generate all 8 figures
used in Section 4 of the BDA project report:
- Figure 4.1: Class distribution (Low, Medium, High)
- Figure 4.2: Congestion by hour of day (Two-peak profile)
- Figure 4.3: Junction hot-spots ranking
- Figure 4.4: Context effects (Day type, Weather, Events)
- Figure 4.5: Model comparison across Accuracy, Precision, Recall, Macro F1
- Figure 4.6: Confusion matrix of Gradient Boosting
- Figure 4.7: Random Forest feature importances
- Figure 4.8: Actual vs Predicted timeline for IT Park (23 September 2026)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration for clean, professional academic look
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists("results.json"):
        raise FileNotFoundError("results.json not found! Please run python train.py first.")
    with open("results.json", "r") as f:
        res = json.load(f)
    return res

def plot_fig1_class_dist(res):
    """Figure 4.1: Distribution of the congestion level 30 minutes ahead."""
    counts = res["summary"]["class_distribution"]
    total = sum(counts.values())
    labels = ["Low", "Medium", "High"]
    vals = [counts[k] for k in labels]
    pcts = [v / total * 100 for v in vals]
    colors = ["#2e7d32", "#f59e0b", "#dc2626"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, pcts, color=colors, width=0.55, edgecolor="none", zorder=3)
    ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Share of records (%)", fontsize=11)
    ax.set_xlabel("Congestion level 30 minutes ahead", fontsize=11)
    ax.set_ylim(0, 100)
    
    for bar, val, pct in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:,}\n({pct:.1f} %)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_class_dist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig2_hourly_profile(res):
    """Figure 4.2: Share of records with congestion, by hour of day."""
    hours = res["hourly_stats"]["hours"]
    high_pct = res["hourly_stats"]["high_pct"]
    med_high_pct = res["hourly_stats"]["med_high_pct"]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, med_high_pct, color="#d97706", marker="o", linewidth=2.2, label="Medium or High", zorder=3)
    ax.plot(hours, high_pct, color="#b91c1c", marker="s", linewidth=2.2, label="High only", zorder=3)
    
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Share of records (%)", fontsize=11)
    ax.set_ylim(-2, 70)
    ax.grid(True, zorder=0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_hourly_profile.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig3_junction_hotspots(res):
    """Figure 4.3: Share of High congestion at each junction."""
    junc_data = res["junction_stats"]
    # Sort lowest to highest for horizontal bar chart
    junc_data = sorted(junc_data, key=lambda x: x["high_pct"])
    names = [x["junction"] for x in junc_data]
    pcts = [x["high_pct"] for x in junc_data]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(names, pcts, color="#2563eb", height=0.55, zorder=3)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("Share of records with High congestion (%)", fontsize=11)
    ax.set_ylabel("Junction", fontsize=11)
    ax.set_xlim(0, max(pcts) * 1.2)
    
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_junction_hotspots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig4_context_effects(res):
    """Figure 4.4: High-congestion share under different conditions."""
    ctx = res["context_stats"]
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    # 1. Day Type
    ax = axes[0]
    bars = ax.bar(["Weekday", "Weekend"], [ctx["weekend"]["without"], ctx["weekend"]["with"]],
                  color=["#64748b", "#cbd5e1"], width=0.5, zorder=3)
    ax.set_title("Day type", fontsize=11, fontweight="bold")
    ax.set_ylabel("High congestion (%)", fontsize=10)
    ax.set_ylim(0, 13)
    ax.grid(axis="y", zorder=0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 2. Weather
    ax = axes[1]
    bars = ax.bar(["Dry", "Rain"], [ctx["rain"]["without"], ctx["rain"]["with"]],
                  color=["#64748b", "#dc2626"], width=0.5, zorder=3)
    ax.set_title("Weather", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 22)
    ax.grid(axis="y", zorder=0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 3. Evening Event
    ax = axes[2]
    bars = ax.bar(["No event", "Event"], [ctx["event"]["without"], ctx["event"]["with"]],
                  color=["#64748b", "#dc2626"], width=0.5, zorder=3)
    ax.set_title("Evening event (J1, J2)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 42)
    ax.grid(axis="y", zorder=0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig4_context_effects.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig5_model_comparison(res):
    """Figure 4.5: Test-set scores of the baseline and the four models."""
    models_order = ["Persistence baseline", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    short_names = ["Persistence\nbaseline", "Logistic\nRegression", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"]
    
    acc = [res["models"][m]["accuracy"] / 100.0 for m in models_order]
    prec = [res["models"][m]["macro_precision"] for m in models_order]
    rec = [res["models"][m]["macro_recall"] for m in models_order]
    f1 = [res["models"][m]["macro_f1"] for m in models_order]
    
    x = np.arange(len(models_order))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, acc, width, label="Accuracy", color="#1e3a8a", zorder=3)
    ax.bar(x - 0.5 * width, prec, width, label="Macro precision", color="#3b82f6", zorder=3)
    ax.bar(x + 0.5 * width, rec, width, label="Macro recall", color="#d97706", zorder=3)
    ax.bar(x + 1.5 * width, f1, width, label="Macro F1", color="#15803d", zorder=3)
    
    ax.set_ylabel("Score (test set)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", ncol=4, frameon=True, facecolor="white", edgecolor="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig5_model_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig6_confusion_matrix(res):
    """Figure 4.6: Confusion matrix of Gradient Boosting (counts and row percentages)."""
    cm = np.array(res["confusion_matrix"])
    labels = ["Low", "Medium", "High"]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted level", fontsize=11)
    ax.set_ylabel("Actual level", fontsize=11)
    
    # Calculate row-wise percentages
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_pcts = cm / row_sums * 100
    
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            pct = row_pcts[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
            
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig6_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig7_feature_importance(res):
    """Figure 4.7: Ten most important features of the Random Forest model."""
    feat_data = res["feature_importance"][:10]
    # Reverse order for horizontal plot
    feat_data = feat_data[::-1]
    
    features = [x["feature"] for x in feat_data]
    scores = [x["importance"] for x in feat_data]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, scores, color="#15803d", height=0.55, zorder=3)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("Random Forest feature importance", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_xlim(0, max(scores) * 1.15)
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", ha="left", va="center", fontsize=9)
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig7_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def plot_fig8_prediction_timeline(res):
    """Figure 4.8: Actual and predicted congestion level, IT Park, 23 September 2026."""
    timeline = res.get("timeline_data", {})
    if not timeline or "timestamps" not in timeline:
        print("[!] Timeline data missing from results.json, skipping fig8.")
        return
        
    ts = timeline["timestamps"]
    act = timeline["actual"]
    pred = timeline["predicted"]
    
    x_indices = np.arange(len(ts))
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(x_indices, act, where="post", color="#1e293b", linewidth=1.8, label="Actual", zorder=3)
    ax.step(x_indices, pred, where="post", color="#dc2626", linestyle="--", linewidth=1.8, label="Predicted (Gradient Boosting)", zorder=3)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"], fontsize=10)
    ax.set_ylabel("Congestion level", fontsize=11)
    
    # Tick every 3 hours (12 intervals of 15 min = 3 hours)
    tick_locs = list(range(0, len(ts), 12)) + [len(ts) - 1]
    tick_labels = [ts[i] for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_xlabel("Time of day (2026-09-23, J3 IT Park)", fontsize=11)
    
    ax.grid(True, zorder=0)
    ax.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig8_prediction_timeline.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[-] Generated {out_path}")

def main():
    print("=" * 70)
    print(" Generating Visualizations for Traffic Congestion BDA Project")
    print("=" * 70)
    res = load_data()
    plot_fig1_class_dist(res)
    plot_fig2_hourly_profile(res)
    plot_fig3_junction_hotspots(res)
    plot_fig4_context_effects(res)
    plot_fig5_model_comparison(res)
    plot_fig6_confusion_matrix(res)
    plot_fig7_feature_importance(res)
    plot_fig8_prediction_timeline(res)
    print("=" * 70)
    print(f" All 8 figures successfully created in '{OUTPUT_DIR}/' folder!")
    print("=" * 70)

if __name__ == "__main__":
    main()

