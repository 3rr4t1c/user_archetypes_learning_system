import os
import glob
import re
import gc
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm

# Import ARLES modules
from arles.arles import ArchetypeLearner, Action
from arles.visuals import (
    prepare_dataframe,
    filter_users,
    cohens_d,
    interpret_effect_size,
    # ARCHETYPE_COLORS,
)

warnings.filterwarnings("ignore")

# Set aggressive garbage collection
gc.set_threshold(700, 10, 10)

# ============================================================================
# GLOBAL PARAMETERS - MODIFY THESE AS NEEDED
# ============================================================================

# Data directory
DATA_FOLDER = "data/pre_post_bluesky"  # Path to folder containing CSV files

# Output directory
OUTPUT_FOLDER = "results/event_analysis"  # Where to save results

# Event configuration
EVENT_CONFIG = {
    "E1": {"name": "X ban in Brazil", "date": "2024-08-30", "color": "#e74c3c"},
    "E2": {"name": "X re-opening in Brazil", "date": "2024-10-08", "color": "#3498db"},
    "E3": {"name": "US elections (Trump)", "date": "2024-11-05", "color": "#9b59b6"},
    "E4": {"name": "Meta policy shift", "date": "2025-01-06", "color": "#e67e22"},
}

# Filtering thresholds
CONFIDENCE_THRESHOLD = 0.5
ARCHETYPE_SUM_THRESHOLD = 0.3

# Processing settings
CHUNK_SIZE = 100000  # Process CSV in chunks of 100k rows
GC_FREQUENCY = 5  # Force GC every N chunks

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # or 'png'

# Sampling rate for reading less data (execution tests)
SAMPLE_RATE = 0.01

# Style
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = FIGURE_DPI
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 80)
print("PRE-POST EVENT ANALYSIS (MEMORY-OPTIMIZED)")
print("=" * 80)
print(f"Data folder: {DATA_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Events configured: {len(EVENT_CONFIG)}")
print(
    f"Filtering: confidence ≥ {CONFIDENCE_THRESHOLD}, sum ≥ {ARCHETYPE_SUM_THRESHOLD}"
)
print(f"Chunked processing: {CHUNK_SIZE:,} rows per chunk")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def print_memory_usage(label: str = "") -> None:
    """Print current memory usage."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"  💾 Memory usage{(' - ' + label) if label else ''}: {mem_gb:.2f} GB")
    except ImportError:
        pass  # psutil not available, skip


def count_csv_rows(filepath: str) -> int:
    """
    Accurately count CSV rows using csv.reader to handle multi-line fields.

    Args:
        filepath: Path to CSV file

    Returns:
        Number of data rows (excluding header)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return sum(1 for _ in reader)


def discover_event_files(folder: str) -> Dict[str, Dict[str, str]]:
    """
    Discover pre/post CSV files for each event.

    Returns:
        Dict with structure: {
            'E1': {'pre': 'path/to/pre_E1.csv', 'post': 'path/to/post_E1.csv'},
            ...
        }
    """
    event_files: Dict[str, Dict[str, str]] = {}

    # Find all pre files
    pre_pattern = os.path.join(folder, "*_pre_E*.csv")
    pre_files = glob.glob(pre_pattern)

    print(f"\nSearching for files in: {folder}")
    print("Pattern: *_pre_E*.csv and *_post_E*.csv")
    print(f"Found {len(pre_files)} pre-event files")

    for pre_file in sorted(pre_files):
        # Extract event ID (e.g., E1, E2, E3, E4)
        match = re.search(r"_pre_(E\d+)\.csv$", pre_file)
        if not match:
            print(f"⚠ Warning: Could not parse event ID from {pre_file}")
            continue

        event_id = match.group(1)

        # Find corresponding post file
        post_file = pre_file.replace(f"_pre_{event_id}.csv", f"_post_{event_id}.csv")

        if not os.path.exists(post_file):
            raise FileNotFoundError(
                f"Missing post file for {event_id}!\n"
                f"Expected: {post_file}\n"
                f"Found pre: {pre_file}"
            )

        event_files[event_id] = {
            "pre": pre_file,
            "post": post_file,
        }

        # Show file sizes
        pre_size = os.path.getsize(pre_file) / 1024 / 1024 / 1024
        post_size = os.path.getsize(post_file) / 1024 / 1024 / 1024
        print(f"  ✓ {event_id}: pre={pre_size:.1f}GB, post={post_size:.1f}GB")

    # Check all configured events have files
    for event_id in EVENT_CONFIG.keys():
        if event_id not in event_files:
            raise FileNotFoundError(
                f"Event {event_id} configured but no files found!\n"
                f"Expected files: *_pre_{event_id}.csv and *_post_{event_id}.csv"
            )

    print(f"\n✓ All {len(event_files)} events have paired pre/post files")
    return event_files


def run_arles_chunked(
    filepath: str, event_id: str, period: str
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Run ARLES on large CSV file with chunked processing.

    Args:
        filepath: Path to CSV file
        event_id: Event identifier (E1, E2, etc.)
        period: 'pre' or 'post'

    Returns:
        Dictionary of archetypes from learner.get_archetypes()
    """
    print(f"\n{'─'*60}")
    print(f"Processing {event_id} {period.upper()}")
    print(f"{'─'*60}")
    print(f"File: {Path(filepath).name}")

    learner = ArchetypeLearner()

    # Count total lines properly using csv.reader
    print("Counting rows (this may take a moment)...")
    total_lines = count_csv_rows(filepath)

    print(f"Total rows: {total_lines:,}")
    print_memory_usage("before processing")

    # Process in chunks
    chunk_count = 0
    processed_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        pbar = tqdm(total=total_lines, desc=f"  {event_id} {period}", unit=" rows")

        for row in reader:

            if np.random.rand() < SAMPLE_RATE:

                try:
                    # Process the line immediately
                    action = Action.from_dict(row)
                    learner.process_action(action)
                    processed_count += 1
                    pbar.update(1)  # Update pbar each processed line

                    # Update counter for periodic cleaning
                    chunk_count += 1
                    if chunk_count % CHUNK_SIZE == 0:
                        # Periodic garbage collection (ogni CHUNK_SIZE righe)
                        if (chunk_count // CHUNK_SIZE) % GC_FREQUENCY == 0:
                            gc.collect()

                except Exception:
                    pbar.update(1)  # Update also if current line fails
                    continue

        pbar.close()

    # Get results
    archetypes = learner.get_archetypes()
    stats = learner.get_stats()

    print(f"✓ Processed: {processed_count:,} actions, {stats['total_users']:,} users")
    print_memory_usage("after processing")

    # Clean up learner
    del learner
    gc.collect()

    return archetypes


def compute_comparison_stats(
    df_pre: pd.DataFrame, df_post: pd.DataFrame, event_id: str
) -> pd.DataFrame:
    """Compute statistical comparison between pre and post periods."""
    results = []

    for archetype in ["superspreader", "amplifier", "coordinated"]:
        data_pre = df_pre[archetype].values
        data_post = df_post[archetype].values

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            data_pre, data_post, alternative="two-sided"
        )

        # Effect size
        effect_size = cohens_d(np.array(data_pre), np.array(data_post))

        # Means and medians
        mean_pre = float(np.mean(np.array(data_pre)))
        mean_post = float(np.mean(np.array(data_post)))
        mean_diff = mean_post - mean_pre

        median_pre = float(np.median(np.array(data_pre)))
        median_post = float(np.median(np.array(data_post)))

        results.append(
            {
                "event_id": event_id,
                "event_name": EVENT_CONFIG[event_id]["name"],
                "archetype": archetype,
                "n_pre": len(data_pre),
                "n_post": len(data_post),
                "mean_pre": mean_pre,
                "mean_post": mean_post,
                "mean_diff": mean_diff,
                "median_pre": median_pre,
                "median_post": median_post,
                "cohens_d": effect_size,
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": interpret_effect_size(effect_size),
            }
        )

    return pd.DataFrame(results)


def generate_latex_table(stats_df: pd.DataFrame) -> str:
    """Generate LaTeX table for statistical results."""

    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{llrrrrrl}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Event} & \\textbf{Archetype} & "
    latex += "\\textbf{Pre} & \\textbf{Post} & "
    latex += "\\textbf{$\\Delta$} & \\textbf{Cohen's d} & "
    latex += "\\textbf{p-value} & \\textbf{Effect} \\\\\n"
    latex += "\\midrule\n"

    current_event = None
    for _, row in stats_df.iterrows():
        # Add separator between events
        if current_event != row["event_id"]:
            if current_event is not None:
                latex += "\\midrule\n"
            current_event = row["event_id"]

        event_name = row["event_name"].replace("&", "\\&")
        archetype = row["archetype"].capitalize()

        # Format p-value
        if row["p_value"] < 0.001:
            p_str = "$<$0.001***"
        elif row["p_value"] < 0.01:
            p_str = f"{row['p_value']:.3f}**"
        elif row["p_value"] < 0.05:
            p_str = f"{row['p_value']:.3f}*"
        else:
            p_str = f"{row['p_value']:.3f}"

        latex += f"{event_name} & {archetype} & "
        latex += f"{row['mean_pre']:.3f} & {row['mean_post']:.3f} & "
        latex += f"{row['mean_diff']:+.3f} & {row['cohens_d']:+.3f} & "
        latex += f"{p_str} & {row['interpretation']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Statistical comparison of archetype scores between "
    latex += "pre-event and post-event periods. "
    latex += "$\\Delta$ indicates mean difference (post - pre). "
    latex += "Significance levels: * p$<$0.05, ** p$<$0.01, *** p$<$0.001. "
    latex += f"Users filtered with confidence $\\geq$ {CONFIDENCE_THRESHOLD} "
    latex += f"and sum(archetypes) $\\geq$ {ARCHETYPE_SUM_THRESHOLD}.}}\n"
    latex += "\\label{tab:event_comparison}\n"
    latex += "\\end{table}\n"

    return latex


# def create_shift_magnitude_plot(
#     df_pre_filt: pd.DataFrame,
#     df_post_filt: pd.DataFrame,
#     event_id: str,
#     event_info: Dict,
# ) -> None:
#     """Create and save shift magnitude plot, ensuring proper cleanup."""
#     common_users = set(df_pre_filt["user_id"]) & set(df_post_filt["user_id"])

#     if len(common_users) < 10:
#         print(f"  ⚠ Too few common users ({len(common_users)}), skipping")
#         return

#     print("\nGenerating shift magnitude plot...")

#     # Compute shifts
#     shifts = []
#     for user_id in common_users:
#         vec_pre = df_pre_filt[df_pre_filt["user_id"] == user_id][
#             ["superspreader", "amplifier", "coordinated"]
#         ].values[0]
#         vec_post = df_post_filt[df_post_filt["user_id"] == user_id][
#             ["superspreader", "amplifier", "coordinated"]
#         ].values[0]

#         shift_magnitude = np.linalg.norm(vec_post - vec_pre)
#         shifts.append(shift_magnitude)

#     shifts_array = np.array(shifts)

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 6))

#     try:
#         ax.hist(
#             shifts_array,
#             bins=50,
#             color=event_info["color"],
#             alpha=0.7,
#             edgecolor="black",
#         )
#         ax.axvline(
#             float(np.mean(shifts_array)),
#             color="red",
#             linestyle="--",
#             linewidth=2.5,
#             label=f"Mean: {np.mean(shifts_array):.3f}",
#         )
#         ax.axvline(
#             float(np.median(shifts_array)),
#             color="orange",
#             linestyle="--",
#             linewidth=2.5,
#             label=f"Median: {np.median(shifts_array):.3f}",
#         )

#         ax.set_xlabel(
#             "Shift magnitude (euclidean distance)", fontweight="bold", fontsize=11
#         )
#         ax.set_ylabel("Number of users", fontweight="bold", fontsize=11)
#         ax.set_title(
#             f"Archetype vector shifts: {event_info['name']}",
#             fontweight="bold",
#             fontsize=13,
#             pad=15,
#         )

#         ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fontsize=10)
#         ax.grid(alpha=0.3, axis="y")

#         # Stats box
#         stats_text = f"N = {len(common_users)} users\n"
#         stats_text += f"Mean shift: {np.mean(shifts_array):.3f}\n"
#         stats_text += f"Median shift: {np.median(shifts_array):.3f}\n"
#         stats_text += f"Max shift: {np.max(shifts_array):.3f}"
#         ax.text(
#             0.5,
#             0.97,
#             stats_text,
#             transform=ax.transAxes,
#             fontsize=10,
#             verticalalignment="top",
#             horizontalalignment="center",
#             bbox=dict(
#                 boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="black"
#             ),
#         )

#         plt.tight_layout()

#         # Save
#         save_path = os.path.join(
#             OUTPUT_FOLDER, f"shift_magnitude_{event_id}.{FIGURE_FORMAT}"
#         )
#         plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
#         print(f"  ✓ Saved: {Path(save_path).name}")

#     finally:
#         # Always clean up
#         plt.close(fig)
#         del fig, ax, shifts, shifts_array
#         gc.collect()


def create_shift_magnitude_plot(
    df_pre_filt: pd.DataFrame,
    df_post_filt: pd.DataFrame,
    event_id: str,
    event_info: Dict,
) -> None:
    """
    Create and save shift magnitude plot.
    Includes sampling of common users for performance.
    """

    # 1. Trova gli utenti comuni e recupera la soglia di campionamento
    common_users = set(df_pre_filt["user_id"]) & set(df_post_filt["user_id"])
    n_common_users = len(common_users)

    # Usa MAX_USERS_FOR_PLOT dalla sezione Global Parameters
    MAX_USERS_FOR_PLOT = globals().get("MAX_USERS_FOR_PLOT", 50000)

    if n_common_users < 10:
        print(f"  ⚠ Too few common users ({n_common_users}), skipping plot.")
        return

    # 2. IMPLEMENTAZIONE DEL CAMPIONAMENTO
    is_sampled = False
    if n_common_users > MAX_USERS_FOR_PLOT:
        is_sampled = True
        print(
            f"  ⚠️ Sampling common users (N={n_common_users:,} -> {MAX_USERS_FOR_PLOT:,}) for plot speed."
        )

        # Campionamento
        common_users_list = list(common_users)
        users_to_process = np.random.choice(
            common_users_list, size=MAX_USERS_FOR_PLOT, replace=False
        )
    else:
        # Processa tutti gli utenti comuni
        users_to_process = list(common_users)

    print(
        f"\nGenerating shift magnitude plot (Processing N={len(users_to_process):,} users)..."
    )

    # 3. Pre-filtra i DataFrame per velocizzare la ricerca interna al loop
    df_pre_filtered = df_pre_filt[df_pre_filt["user_id"].isin(users_to_process)]
    df_post_filtered = df_post_filt[df_post_filt["user_id"].isin(users_to_process)]

    # 4. Compute shifts
    shifts = []

    # Ciclo con progress bar
    for user_id in tqdm(users_to_process, desc="  Computing shifts"):
        try:
            # Ricerca molto più veloce grazie alla pre-filtrazione
            vec_pre = df_pre_filtered[df_pre_filtered["user_id"] == user_id][
                ["superspreader", "amplifier", "coordinated"]
            ].values[0]
            vec_post = df_post_filtered[df_post_filtered["user_id"] == user_id][
                ["superspreader", "amplifier", "coordinated"]
            ].values[0]

            shift_magnitude = np.linalg.norm(vec_post - vec_pre)
            shifts.append(shift_magnitude)
        except IndexError:
            # Se l'utente non è trovato nel DataFrame filtrato (non dovrebbe accadere con isin), lo saltiamo
            continue

    shifts_array = np.array(shifts)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        ax.hist(
            shifts_array,
            bins=50,
            color=event_info["color"],
            alpha=0.7,
            edgecolor="black",
        )
        ax.axvline(
            float(np.mean(shifts_array)),
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Mean: {np.mean(shifts_array):.3f}",
        )
        ax.axvline(
            float(np.median(shifts_array)),
            color="orange",
            linestyle="--",
            linewidth=2.5,
            label=f"Median: {np.median(shifts_array):.3f}",
        )

        ax.set_xlabel(
            "Shift magnitude (euclidean distance)", fontweight="bold", fontsize=11
        )
        ax.set_ylabel("Number of users", fontweight="bold", fontsize=11)
        ax.set_title(
            f"Archetype vector shifts: {event_info['name']}",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )

        ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # Stats box aggiornato per riflettere il campionamento
        stats_text = f"N = {len(shifts_array):,} users"
        if is_sampled:
            stats_text += f" (Sampled from {n_common_users:,})"
        stats_text += f"\nMean shift: {np.mean(shifts_array):.3f}\n"
        stats_text += f"Median shift: {np.median(shifts_array):.3f}\n"
        stats_text += f"Max shift: {np.max(shifts_array):.3f}"

        ax.text(
            0.5,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="black"
            ),
        )

        plt.tight_layout()

        # Save
        save_path = os.path.join(
            OUTPUT_FOLDER, f"shift_magnitude_{event_id}.{FIGURE_FORMAT}"
        )
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  ✓ Saved: {Path(save_path).name}")

    finally:
        # Always clean up
        plt.close(fig)
        del fig, ax, shifts, shifts_array
        gc.collect()


# ============================================================================
# STEP 1: DISCOVER FILES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DISCOVERING DATA FILES")
print("=" * 80)

event_files = discover_event_files(DATA_FOLDER)
print_memory_usage("initial")

# ============================================================================
# STEP 2-6: PROCESS EACH EVENT INDEPENDENTLY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2-6: PROCESSING EVENTS (ONE AT A TIME)")
print("=" * 80)

# Storage for results
all_stats: List[pd.DataFrame] = []
event_means: Dict[str, Dict[str, Dict[str, float]]] = {}

for event_id in sorted(EVENT_CONFIG.keys()):
    event_info = EVENT_CONFIG[event_id]

    print(f"\n{'='*80}")
    print(f"EVENT {event_id}: {event_info['name']}")
    print(f"{'='*80}")

    # Step 2: Run ARLES on pre and post
    archetypes_pre = run_arles_chunked(event_files[event_id]["pre"], event_id, "pre")
    archetypes_post = run_arles_chunked(event_files[event_id]["post"], event_id, "post")

    # Step 3: Prepare DataFrames
    print("\nPreparing DataFrames...")
    df_pre = prepare_dataframe(archetypes_pre)
    df_post = prepare_dataframe(archetypes_post)

    print("\nFiltering users (pre):")
    df_pre_filt = filter_users(
        df_pre,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sum_threshold=ARCHETYPE_SUM_THRESHOLD,
        use_sum=True,
        verbose=True,
    )

    print("\nFiltering users (post):")
    df_post_filt = filter_users(
        df_post,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sum_threshold=ARCHETYPE_SUM_THRESHOLD,
        use_sum=True,
        verbose=True,
    )

    # Store means for evolution plot
    event_means[event_id] = {
        "pre": {
            "superspreader": float(df_pre_filt["superspreader"].mean()),
            "amplifier": float(df_pre_filt["amplifier"].mean()),
            "coordinated": float(df_pre_filt["coordinated"].mean()),
        },
        "post": {
            "superspreader": float(df_post_filt["superspreader"].mean()),
            "amplifier": float(df_post_filt["amplifier"].mean()),
            "coordinated": float(df_post_filt["coordinated"].mean()),
        },
    }

    # Step 4: Compute statistics
    print("\nComputing statistics...")
    stats_df = compute_comparison_stats(df_pre_filt, df_post_filt, event_id)
    all_stats.append(stats_df)

    # Print summary
    for _, row in stats_df.iterrows():
        sig = (
            "***"
            if row["p_value"] < 0.001
            else "**" if row["p_value"] < 0.01 else "*" if row["significant"] else "ns"
        )
        print(
            f"  {row['archetype']:15s}: "
            f"Δ = {row['mean_diff']:+.3f} "
            f"(d = {row['cohens_d']:+.3f}, {row['interpretation']}, {sig})"
        )

    # Step 5: Shift magnitude plot (with proper cleanup)
    create_shift_magnitude_plot(df_pre_filt, df_post_filt, event_id, event_info)

    # Step 6: Clean up memory aggressively
    print("\n🧹 Cleaning up memory...")
    del archetypes_pre, archetypes_post
    del df_pre, df_post, df_pre_filt, df_post_filt, stats_df
    gc.collect()
    print_memory_usage("after cleanup")

    print(f"\n✓ {event_id} complete")

# ============================================================================
# STEP 7: AGGREGATE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: AGGREGATING STATISTICS")
print("=" * 80)

all_stats_df = pd.concat(all_stats, ignore_index=True)

# Save to CSV
stats_csv_path = os.path.join(OUTPUT_FOLDER, "statistical_comparison.csv")
all_stats_df.to_csv(stats_csv_path, index=False)
print(f"✓ Statistics saved to: {stats_csv_path}")

# Generate LaTeX table
latex_table = generate_latex_table(all_stats_df)
latex_path = os.path.join(OUTPUT_FOLDER, "table_event_comparison.tex")
with open(latex_path, "w") as f:
    f.write(latex_table)
print(f"✓ LaTeX table saved to: {latex_path}")

# ============================================================================
# STEP 8: EVOLUTION PLOT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: CREATING EVOLUTION PLOT")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

try:
    archetypes = ["superspreader", "amplifier", "coordinated"]
    event_ids = sorted(EVENT_CONFIG.keys())

    for col_idx, archetype in enumerate(archetypes):
        ax = axes[col_idx]

        for event_idx, event_id in enumerate(event_ids):
            event_info = EVENT_CONFIG[event_id]

            # Get means
            mean_pre = event_means[event_id]["pre"][archetype]
            mean_post = event_means[event_id]["post"][archetype]

            # x positions
            x_pre = event_idx - 0.15
            x_post = event_idx + 0.15

            # Plot pre point
            ax.scatter(
                x_pre,
                mean_pre,
                s=150,
                color=event_info["color"],
                alpha=0.5,
                edgecolors="black",
                linewidths=1.5,
                marker="o",
                zorder=3,
            )

            # Plot post point
            ax.scatter(
                x_post,
                mean_post,
                s=150,
                color=event_info["color"],
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
                marker="o",
                zorder=3,
            )

            # Connect with arrow
            ax.annotate(
                "",
                xy=(x_post, mean_post),
                xytext=(x_pre, mean_pre),
                arrowprops=dict(
                    arrowstyle="->",
                    color=event_info["color"],
                    lw=2.5,
                    alpha=0.7,
                ),
                zorder=2,
            )

            # Add event label (only on first subplot)
            if col_idx == 0:
                ax.text(
                    event_idx,
                    -0.08,
                    event_info["name"],
                    ha="center",
                    va="top",
                    fontsize=8,
                    rotation=0,
                    transform=ax.get_xaxis_transform(),
                )

        # Styling
        ax.set_xticks(range(len(event_ids)))
        ax.set_xticklabels([f"E{i+1}" for i in range(len(event_ids))])
        ax.set_ylabel(f"{archetype.capitalize()} Score", fontweight="bold", fontsize=12)
        ax.set_title(
            f"({chr(65+col_idx)}) {archetype.capitalize()} Evolution",
            fontweight="bold",
            fontsize=13,
            pad=10,
        )
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)

        # # Add pre/post legend (only on last subplot)
        # if col_idx == 2:
        #     legend_elements = [
        #         Line2D(
        #             [0],
        #             [0],
        #             marker="o",
        #             color="w",
        #             markerfacecolor="gray",
        #             markersize=10,
        #             alpha=0.5,
        #             markeredgecolor="black",
        #             label="Pre-event",
        #         ),
        #         Line2D(
        #             [0],
        #             [0],
        #             marker="o",
        #             color="w",
        #             markerfacecolor="gray",
        #             markersize=10,
        #             alpha=0.9,
        #             markeredgecolor="black",
        #             label="Post-event",
        #         ),
        #     ]
        #     ax.legend(
        #         handles=legend_elements,
        #         loc="upper right",
        #         framealpha=0.95,
        #         fontsize=10,
        #     )

    plt.tight_layout()

    # Comment this and uncomment the above to have legend in just one plot
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            alpha=0.5,
            markeredgecolor="black",
            label="Pre-event",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            alpha=0.9,
            markeredgecolor="black",
            label="Post-event",
        ),
    ]

    # Add legend to the figure not to the specified axis
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        framealpha=0.95,
        fontsize=10,
    )

    evolution_plot_path = os.path.join(
        OUTPUT_FOLDER, f"archetype_evolution.{FIGURE_FORMAT}"
    )
    plt.savefig(evolution_plot_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"✓ Saved: {evolution_plot_path}")

finally:
    # Always clean up
    plt.close(fig)
    del fig, axes
    gc.collect()

# ============================================================================
# FINAL CLEANUP AND SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files in: {OUTPUT_FOLDER}/")
print(f"  • archetype_evolution.{FIGURE_FORMAT}")
for event_id in EVENT_CONFIG.keys():
    print(f"  • shift_magnitude_{event_id}.{FIGURE_FORMAT}")
print("  • statistical_comparison.csv")
print("  • table_event_comparison.tex")

print_memory_usage("final")

print("\n" + "=" * 80)

# Final aggressive cleanup
del all_stats, all_stats_df, event_means
gc.collect()
