import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

"""
Comparison script for three optimization strategies.
Assumes folder structure:
  /kaggle/output/
      ga_results/
      greedy_results/
      pso_results/
Each of those contains JSON files produced by the enriched notebooks:
  result_nodes_150.json, result_nodes_200.json, ..., up to 550 (as available).

If your actual folder names differ, edit METHOD_FOLDERS below.
Run inside Kaggle or local environment after results are generated.
Outputs:
  combined_metrics.csv
  comparison_summary.txt
  comparison_plots.png
"""

# Configure and auto-detect base output directory and method folders
def _pick_base_dir() -> str:
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.environ.get("COMPARE_BASE"),
        os.path.join(script_dir, "output"),
        "/kaggle/output",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    # Fallback to local under script dir
    return os.path.join(script_dir, "output")

BASE_OUTPUT_DIR = _pick_base_dir()

# Fixed folder names for each method
METHOD_FOLDERS = {
    "GA": "ga",
    "Greedy": "greedy_exact_method",
    "PSO": "pso",
}

def _resolve_method_folder(method: str, base_dir: str, configured_name: str) -> str | None:
    """Resolve the folder path for a method - only use the configured name"""
    path = os.path.join(base_dir, configured_name)
    if os.path.isdir(path):
        return path
    return None

# Regex to extract node count (digits) from filename
NODE_RE = re.compile(r"(\d+)")

# Metrics to extract (keys map to lambda for safe retrieval)
EXTRACTION_FUNCS = {
    "initial_total_nodes": lambda m: m.get("nodes_summary", {}).get("initial_total_nodes"),
    "final_alive_nodes": lambda m: m.get("nodes_summary", {}).get("final_alive_nodes"),
    "final_alive_ratio": lambda m: m.get("nodes_summary", {}).get("final_alive_ratio"),
    "total_dead_nodes": lambda m: m.get("nodes_summary", {}).get("total_dead_nodes"),
    "initial_best_time": lambda m: m.get("path_stats", {}).get("initial_best_time"),
    "final_best_time": lambda m: m.get("path_stats", {}).get("final_best_time"),
    "final_best_distance": lambda m: m.get("path_stats", {}).get("final_best_distance"),
    "cycles_completed": lambda m: m.get("lifecycle", {}).get("cycles_completed"),
    "reoptimization_count": lambda m: m.get("lifecycle", {}).get("reoptimization_count"),
    "recluster_events": lambda m: len(m.get("lifecycle", {}).get("reclustering_cycles", [])),
    "initial_num_clusters": lambda m: m.get("clustering_initial", {}).get("num_clusters"),
    "final_num_clusters": lambda m: m.get("clustering_final", {}).get("num_clusters"),
    "final_total_energy": lambda m: m.get("energy", {}).get("final_total_energy"),
    "initial_total_energy": lambda m: m.get("energy", {}).get("initial_total_energy"),
    "first_death_cycle": lambda m: m.get("death_analytics", {}).get("first_death_cycle"),
    "last_death_cycle": lambda m: m.get("death_analytics", {}).get("last_death_cycle"),
    "death_events_count": lambda m: len(m.get("death_events", [])),
    "mean_deaths_per_event": lambda m: m.get("death_analytics", {}).get("mean_deaths_per_event"),
}

def extract_node_count(filename: str) -> int | None:
    match = NODE_RE.findall(filename)
    if not match:
        return None
    # choose the largest found number (in case of multiple numbers) as node count
    nums = [int(x) for x in match]
    return max(nums) if nums else None

def load_method_results(method: str, folder_name: str) -> List[Dict[str, Any]]:
    folder_path = _resolve_method_folder(method, BASE_OUTPUT_DIR, folder_name)
    if not folder_path:
        print(f"[WARN] Missing folder for method {method} in base {BASE_OUTPUT_DIR} (tried variants of '{folder_name}')")
        return []
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.json'):
            continue
        # Accept any json; many runs may not prefix with 'result_'
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")
            continue
        node_count = extract_node_count(fname)
        if node_count is None:
            # try from meta
            node_count = meta.get('nodes_summary', {}).get('initial_total_nodes')
        record = {
            'method': method,
            'filename': fname,
            'nodes': node_count
        }
        for col, func in EXTRACTION_FUNCS.items():
            try:
                record[col] = func(meta)
            except Exception:
                record[col] = None
        # Derived metrics
        initial_nodes = record.get('initial_total_nodes') or 0
        dead_nodes = record.get('total_dead_nodes') or 0
        cycles = record.get('cycles_completed') or 0
        record['death_rate'] = dead_nodes / initial_nodes if initial_nodes else None
        record['avg_dead_per_cycle'] = dead_nodes / cycles if cycles else None
        results.append(record)
    return results

def build_dataframe() -> pd.DataFrame:
    all_records = []
    for method, folder in METHOD_FOLDERS.items():
        all_records.extend(load_method_results(method, folder))
    df = pd.DataFrame(all_records)
    # Sort by nodes then method for readability
    if 'nodes' in df.columns:
        df = df.sort_values(['nodes', 'method'])
    return df

def summarize(df: pd.DataFrame) -> str:
    lines = []
    if df.empty:
        lines.append("No data loaded.")
        return "\n".join(lines)
    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method]
        lines.append(f"\n-- {method} --")
        lines.append(f"Files: {len(sub)} | Node range: {sub['nodes'].min()} - {sub['nodes'].max()}")
        for metric in [
            'final_best_time', 'cycles_completed',
            'death_rate', 'avg_dead_per_cycle', 'reoptimization_count', 'recluster_events'
        ]:
            if metric in sub.columns:
                val_mean = sub[metric].dropna().mean()
                lines.append(f"Mean {metric}: {val_mean:.4f}")
    # Identify best method per node for final_best_time if available
    if 'final_best_time' in df.columns and 'nodes' in df.columns:
        lines.append("\nBest method per node (min final_best_time):")
        for node in sorted(df['nodes'].dropna().unique()):
            slice_df = df[df['nodes'] == node]
            best_row = slice_df.loc[slice_df['final_best_time'].idxmin()] if not slice_df['final_best_time'].isna().all() else None
            if best_row is not None:
                lines.append(f"  Nodes {node}: {best_row['method']} ({best_row['final_best_time']:.3f}s)")
    return "\n".join(lines)

def plot_metrics(df: pd.DataFrame, output_path: str):
    if df.empty:
        print("[INFO] Nothing to plot; DataFrame empty.")
        return
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    # Hide the last empty subplot (axes[2,1])
    axes[2,1].set_visible(False)

    # Panel 1: Final best time vs nodes
    ax = axes[0,0]
    sns.lineplot(data=df, x='nodes', y='final_best_time', hue='method', marker='o', ax=ax)
    ax.set_title('Thời gian tốt nhất đạt được')
    ax.set_ylabel('Time (s)')

    # Panel 2: Cycles completed vs nodes
    ax = axes[0,1]
    sns.lineplot(data=df, x='nodes', y='cycles_completed', hue='method', marker='^', ax=ax)
    ax.set_title('Số Cycle đã hoàn thành')
    ax.set_ylabel('Cycles')

    # Panel 3: Death rate vs nodes
    ax = axes[1,0]
    sns.lineplot(data=df, x='nodes', y='death_rate', hue='method', marker='d', ax=ax)
    ax.set_title('Tỷ lệ node chết')
    ax.set_ylabel('Dead / Initial')

    # Panel 4: Number of clusters vs nodes
    ax = axes[1,1]
    sns.lineplot(data=df, x='nodes', y='final_num_clusters', hue='method', marker='*', ax=ax)
    ax.set_title('Số cụm cuối cùng theo')
    ax.set_ylabel('Number of Clusters')
    ax.set_xlabel('Nodes')

    # Panel 5: Final average energy vs nodes
    ax = axes[2,0]
    sns.lineplot(data=df, x='nodes', y='final_total_energy', hue='method', marker='x', ax=ax)
    ax.set_title('Năng lượng trung bình cuối cùng')
    ax.set_ylabel('Average Energy')
    ax.set_xlabel('Nodes')

    for ax in axes.flatten():
        ax.legend(fontsize=8)
        ax.set_xlabel('Nodes')

    plt.suptitle('WSN Optimization Method Comparison', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Plots saved to {output_path}")

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Using base output dir: {BASE_OUTPUT_DIR}")
    df = build_dataframe()
    print("Loaded records:", len(df))
    summary_text = summarize(df)
    print(summary_text)

    # Save artifacts
    combined_csv = os.path.join(BASE_OUTPUT_DIR, 'combined_metrics.csv')
    df.to_csv(combined_csv, index=False)
    print(f"[OK] CSV written: {combined_csv} Total columns: {len(df.columns)}")

    summary_path = os.path.join(BASE_OUTPUT_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"[OK] Summary written: {summary_path}")

    plot_path = os.path.join(BASE_OUTPUT_DIR, 'comparison_plots.png')
    plot_metrics(df, plot_path)

    # Optional: pivot tables for quick glance
    if not df.empty:
        pivot_time = df.pivot_table(index='nodes', columns='method', values='final_best_time')
        print("\nFinal Best Time (pivot):")
        print(pivot_time)

if __name__ == '__main__':
    main()
