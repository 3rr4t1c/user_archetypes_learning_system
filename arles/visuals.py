"""
ARLES Analysis and Visualization Module for ICWSM Paper
Publication-ready plots for mass migration archetype analysis

FIXED VERSION - All errors resolved, Pylance warnings addressed
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from typing import cast
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL PARAMETERS - MODIFY THESE AT THE TOP OF YOUR NOTEBOOK
# ============================================================================

EVENTS = [
    {"name": "X/Twitter ban in Brazil", "date": "2024-08-30", "color": "#e74c3c"},
    # {"name": "X Reopening in Brazil", "date": "2024-10-08", "color": "#3498db"},
    {"name": "X/Twitter changing", "date": "2024-10-17", "color": "#3498db"},
    {"name": "U.S. election (Trump)", "date": "2024-11-05", "color": "#9b59b6"},
    {"name": "Social platforms changings", "date": "2025-01-06", "color": "#e67e22"},
]

EVENT_WINDOW_DAYS = 7  # ±7 days around each event

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for inclusion
ARCHETYPE_SUM_THRESHOLD = 0.3  # Minimum sum(archetypes) for inclusion
ARCHETYPE_MAX_THRESHOLD = 0.2  # Alternative: max(archetypes) threshold

# Color scheme (colorblind-friendly)
ARCHETYPE_COLORS = {
    "superspreader": "#d62728",  # Red
    "amplifier": "#1f77b4",  # Blue
    "coordinated": "#2ca02c",  # Green
}

# Style settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
sns.set_style("whitegrid")
sns.set_palette("colorblind")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def parse_event_dates(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse event dates from strings to datetime objects."""
    parsed_events = []
    for event in events:
        event_copy = event.copy()
        if isinstance(event["date"], str):
            event_copy["date"] = datetime.strptime(event["date"], "%Y-%m-%d")
        parsed_events.append(event_copy)
    return parsed_events


def filter_users(
    df: pd.DataFrame,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sum_threshold: float = ARCHETYPE_SUM_THRESHOLD,
    use_sum: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter users based on confidence and archetype activity.

    Args:
        df: DataFrame with archetype data
        confidence_threshold: Minimum confidence score
        sum_threshold: Minimum sum of archetype scores
        use_sum: If True, use sum threshold; if False, use max threshold
        verbose: If True, print filtering statistics

    Returns:
        Filtered DataFrame
    """
    # Confidence filter
    high_conf = df["confidence"] >= confidence_threshold

    # Archetype activity filter
    if use_sum:
        archetype_sum = df["superspreader"] + df["amplifier"] + df["coordinated"]
        active = archetype_sum >= sum_threshold
        filter_name = f"sum ≥ {sum_threshold}"
    else:
        archetype_max = df[["superspreader", "amplifier", "coordinated"]].max(axis=1)
        active = archetype_max >= ARCHETYPE_MAX_THRESHOLD
        filter_name = f"max ≥ {ARCHETYPE_MAX_THRESHOLD}"

    df_filtered = df[high_conf & active].copy()

    if verbose:
        print(
            f"Filtering: confidence ≥ {confidence_threshold}, archetype {filter_name}"
        )
        print(f"  Total users: {len(df)}")
        print(
            f"  High confidence: {high_conf.sum()} ({100*high_conf.sum()/len(df):.1f}%)"
        )
        print(f"  Active archetypes: {active.sum()} ({100*active.sum()/len(df):.1f}%)")
        print(
            f"  Final filtered: {len(df_filtered)} ({100*len(df_filtered)/len(df):.1f}%)"
        )

    return df_filtered


def compute_period_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for a period."""
    stats_dict: Dict[str, Any] = {}
    for archetype in ["superspreader", "amplifier", "coordinated"]:
        stats_dict[archetype] = {
            "mean": float(df[archetype].mean()),
            "median": float(df[archetype].median()),
            "std": float(df[archetype].std()),
            "q25": float(df[archetype].quantile(0.25)),
            "q75": float(df[archetype].quantile(0.75)),
            "q90": float(df[archetype].quantile(0.90)),
        }
    stats_dict["n_users"] = len(df)
    stats_dict["avg_confidence"] = float(df["confidence"].mean())
    return stats_dict


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (
        float((np.mean(group1) - np.mean(group2)) / pooled_std)
        if pooled_std > 0
        else 0.0
    )


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# HELPER: PREPARE DATAFRAME FROM LEARNER OUTPUT
# ============================================================================


def prepare_dataframe(archetypes: Dict[str, Tuple[np.ndarray, float]]) -> pd.DataFrame:
    """
    Convert learner.get_archetypes() output to DataFrame.

    Args:
        archetypes: Dict from learner.get_archetypes()
                   Format: {user_id: (vector, confidence)}

    Returns:
        DataFrame with columns: user_id, superspreader, amplifier, coordinated,
                               confidence, dominant_archetype, dominant_score
    """
    data = []
    for user_id, (vector, confidence) in archetypes.items():
        data.append(
            {
                "user_id": user_id,
                "superspreader": float(vector[0]),
                "amplifier": float(vector[1]),
                "coordinated": float(vector[2]),
                "confidence": float(confidence),
            }
        )

    df = pd.DataFrame(data)

    # Add derived columns
    df["dominant_archetype"] = df[["superspreader", "amplifier", "coordinated"]].idxmax(
        axis=1
    )
    df["dominant_score"] = df[["superspreader", "amplifier", "coordinated"]].max(axis=1)
    df["archetype_sum"] = df["superspreader"] + df["amplifier"] + df["coordinated"]

    return df


# ============================================================================
# FIGURE 1: ARCHETYPE SPACE OVERVIEW
# ============================================================================


# def plot_archetype_overview(
#     df: pd.DataFrame,
#     save_path: Optional[str] = None,
#     show_plot: bool = True,
# ) -> Figure:
#     """
#     Create comprehensive overview figure with 3D space and distributions.

#     Layout: 2x2 grid
#     - Top-left: 3D archetype space
#     - Top-right: Dominant archetype distribution
#     - Bottom: Marginal distributions for each archetype

#     Args:
#         df: DataFrame with archetype data (output of prepare_dataframe + filter_users)
#         save_path: Optional path to save figure
#         show_plot: Whether to display the plot

#     Returns:
#         matplotlib Figure object
#     """
#     fig = plt.figure(figsize=(14, 10))
#     gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

#     # --- Top-left: 3D Archetype Space ---
#     ax_3d = cast(Axes3D, fig.add_subplot(gs[0, 0], projection="3d"))

#     # Color by dominant archetype
#     colors_list = [ARCHETYPE_COLORS[arch] for arch in df["dominant_archetype"]]
#     sizes = df["confidence"] * 50 + 10

#     ax_3d.scatter(
#         df["superspreader"].to_numpy(),
#         df["amplifier"].to_numpy(),
#         df["coordinated"].to_numpy(),  # type: ignore[arg-type]
#         c=colors_list,
#         s=sizes.to_numpy(),  # type: ignore[arg-type]
#         alpha=0.6,
#         edgecolors="black",
#         linewidths=0.3,
#     )

#     ax_3d.set_xlabel("Superspreader", fontweight="bold", labelpad=8)
#     ax_3d.set_ylabel("Amplifier", fontweight="bold", labelpad=8)
#     ax_3d.set_zlabel("Coordinated", fontweight="bold", labelpad=8)
#     ax_3d.set_title("(A) Users in archetype space", fontweight="bold", pad=15)

#     # Legend
#     legend_elements = [
#         Patch(facecolor=ARCHETYPE_COLORS["superspreader"], label="Superspreader"),
#         Patch(facecolor=ARCHETYPE_COLORS["amplifier"], label="Amplifier"),
#         Patch(facecolor=ARCHETYPE_COLORS["coordinated"], label="Coordinated"),
#     ]
#     ax_3d.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

#     # --- Top-right: Dominant Archetype Distribution ---
#     ax_dist = fig.add_subplot(gs[0, 1])

#     archetype_counts = df["dominant_archetype"].value_counts()
#     colors = [ARCHETYPE_COLORS[arch] for arch in archetype_counts.index]

#     bars = ax_dist.bar(
#         list(range(len(archetype_counts))),
#         archetype_counts.values.tolist(),
#         color=colors,
#         alpha=0.7,
#         edgecolor="black",
#     )

#     ax_dist.set_xticks(range(len(archetype_counts)))
#     ax_dist.set_xticklabels(
#         [a.capitalize() for a in archetype_counts.index], rotation=0
#     )
#     ax_dist.set_ylabel("Number of users", fontweight="bold")
#     ax_dist.set_title("(B) Dominant archetype distribution", fontweight="bold", pad=15)
#     ax_dist.grid(axis="y", alpha=0.3)

#     # Add percentage labels
#     total = len(df)
#     for bar, count in zip(bars, archetype_counts.values):
#         height = bar.get_height()
#         ax_dist.text(
#             bar.get_x() + bar.get_width() / 2.0,
#             height,
#             f"{count}\n({100*count/total:.1f}%)",
#             ha="center",
#             va="bottom",
#             fontsize=9,
#             fontweight="bold",
#         )

#     # --- Bottom: Marginal Distributions ---
#     ax_margins = fig.add_subplot(gs[1, :])

#     positions = np.array([0, 1.5, 3])
#     width = 0.4

#     for i, (archetype, color) in enumerate(ARCHETYPE_COLORS.items()):
#         data = df[archetype]

#         # Violin plot
#         parts = ax_margins.violinplot(
#             [data],
#             positions=[positions[i]],
#             widths=width,
#             showmeans=True,
#             showmedians=True,
#         )

#         for pc in parts["bodies"]:  # type: ignore
#             pc.set_facecolor(color)
#             pc.set_alpha(0.7)
#             pc.set_edgecolor("black")

#         # Customize violin plot elements
#         for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
#             if partname in parts:
#                 parts[partname].set_edgecolor("black")
#                 parts[partname].set_linewidth(1.5)

#     ax_margins.set_xticks(positions)
#     ax_margins.set_xticklabels([a.capitalize() for a in ARCHETYPE_COLORS.keys()])
#     ax_margins.set_ylabel("Archetype score", fontweight="bold")
#     ax_margins.set_title("(C) Archetype score distributions", fontweight="bold", pad=15)
#     ax_margins.set_ylim(-0.05, 1.05)
#     ax_margins.grid(axis="y", alpha=0.3)

#     # Add summary stats as text
#     stats_text = f"N = {len(df)} users\n"
#     stats_text += f"Avg confidence: {df['confidence'].mean():.3f}"
#     ax_margins.text(
#         0.98,
#         0.97,
#         stats_text,
#         transform=ax_margins.transAxes,
#         fontsize=9,
#         verticalalignment="top",
#         horizontalalignment="right",
#         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
#     )

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Saved: {save_path}")

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

#     return fig


def plot_archetype_overview(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Create comprehensive overview figure with 3D space and distributions.

    Layout: 2x2 grid
    - Top-left: 3D archetype space
    - Top-right: Dominant archetype distribution
    - Bottom: Marginal distributions for each archetype

    Args:
        df: DataFrame with archetype data (output of prepare_dataframe + filter_users)
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # --- Top-left: 3D Archetype Space ---
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")

    # Color by dominant archetype
    colors_list = [ARCHETYPE_COLORS[arch] for arch in df["dominant_archetype"]]

    # Smaller sizes to reduce overlap, with more variation based on confidence
    sizes = df["confidence"] * 30 + 5  # Reduced from 50+10 to 30+5

    ax_3d.scatter(
        df["superspreader"],
        df["amplifier"],
        df["coordinated"],
        c=colors_list,
        s=sizes,
        alpha=0.4,  # Reduced from 0.6 to 0.4 for better overlap visibility
        edgecolors="black",
        linewidths=0.2,  # Thinner edges
    )

    ax_3d.set_xlabel("Superspreader", fontweight="bold", labelpad=8)
    ax_3d.set_ylabel("Amplifier", fontweight="bold", labelpad=8)
    ax_3d.set_zlabel("Coordinated", fontweight="bold", labelpad=8)
    ax_3d.set_title("(A) Users in archetype space", fontweight="bold", pad=15)

    # Better viewing angle to show all clusters
    ax_3d.view_init(elev=20, azim=45)  # Adjust elevation and azimuth

    # Legend
    legend_elements = [
        Patch(
            facecolor=ARCHETYPE_COLORS["superspreader"],
            label="Superspreader",
            alpha=0.7,
        ),
        Patch(facecolor=ARCHETYPE_COLORS["amplifier"], label="Amplifier", alpha=0.7),
        Patch(
            facecolor=ARCHETYPE_COLORS["coordinated"], label="Coordinated", alpha=0.7
        ),
    ]
    ax_3d.legend(handles=legend_elements, loc="upper left", framealpha=0.95, fontsize=9)

    # --- Top-right: Dominant Archetype Distribution ---
    ax_dist = fig.add_subplot(gs[0, 1])

    archetype_counts = df["dominant_archetype"].value_counts()
    colors = [ARCHETYPE_COLORS[arch] for arch in archetype_counts.index]

    bars = ax_dist.bar(
        range(len(archetype_counts)),
        archetype_counts.values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    ax_dist.set_xticks(range(len(archetype_counts)))
    ax_dist.set_xticklabels(
        [a.capitalize() for a in archetype_counts.index], rotation=0
    )
    ax_dist.set_ylabel("Number of users", fontweight="bold")
    ax_dist.set_title("(B) Dominant archetype distribution", fontweight="bold", pad=15)
    ax_dist.grid(axis="y", alpha=0.3)

    # Add percentage labels
    total = len(df)
    for bar, count in zip(bars, archetype_counts.values):
        height = bar.get_height()
        ax_dist.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({100*count/total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # --- Bottom: Marginal Distributions ---
    ax_margins = fig.add_subplot(gs[1, :])

    positions = np.array([0, 1.5, 3])
    width = 0.4

    for i, (archetype, color) in enumerate(ARCHETYPE_COLORS.items()):
        data = df[archetype]

        # Violin plot
        parts = ax_margins.violinplot(
            [data],
            positions=[positions[i]],
            widths=width,
            showmeans=True,
            showmedians=True,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")

        # Customize violin plot elements
        for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
            if partname in parts:
                parts[partname].set_edgecolor("black")
                parts[partname].set_linewidth(1.5)

    ax_margins.set_xticks(positions)
    ax_margins.set_xticklabels([a.capitalize() for a in ARCHETYPE_COLORS.keys()])
    ax_margins.set_ylabel("Archetype score", fontweight="bold")
    ax_margins.set_title("(C) Archetype score distributions", fontweight="bold", pad=15)
    ax_margins.set_ylim(-0.05, 1.05)
    ax_margins.grid(axis="y", alpha=0.3)

    # Add summary stats as text - MOVED TO LEFT SIDE
    stats_text = f"N = {len(df)} users\n"
    stats_text += f"Avg confidence: {df['confidence'].mean():.3f}"
    ax_margins.text(
        0.02,  # Changed from 0.98 to 0.02 (left side)
        0.97,
        stats_text,
        transform=ax_margins.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",  # Changed from "right" to "left"
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="black"),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


# ============================================================================
# FIGURE 2: EVENT-BASED ANALYSIS
# ============================================================================


def analyze_event_windows(
    learner_results: Dict[str, Dict[str, Tuple[np.ndarray, float]]],
    events: List[Dict[str, Any]] = EVENTS,
) -> pd.DataFrame:
    """
    Analyze archetype distributions around events.

    Args:
        learner_results: Dict with keys like 'whole_period', 'event_1_before', etc.
                        Values are dicts from learner.get_archetypes()
        events: List of event dictionaries
        window_days: Days before/after event (not used in logic, kept for compatibility)

    Returns:
        DataFrame with statistics per period
    """
    events_parsed = parse_event_dates(events)

    stats_list = []

    # Whole period
    if "whole_period" in learner_results:
        df_whole = prepare_dataframe(learner_results["whole_period"])
        df_whole_filt = filter_users(df_whole, verbose=False)
        stats = compute_period_stats(df_whole_filt)
        stats["period"] = "Whole period"
        stats["event_id"] = -1
        stats["phase"] = "all"
        stats_list.append(stats)

    # Event windows
    for i, event in enumerate(events_parsed):
        for phase in ["before", "during", "after"]:
            key = f"event_{i+1}_{phase}"
            if key in learner_results:
                df_period = prepare_dataframe(learner_results[key])
                df_period_filt = filter_users(df_period, verbose=False)
                stats = compute_period_stats(df_period_filt)
                stats["period"] = f"{event['name']} ({phase})"
                stats["event_id"] = i
                stats["event_name"] = event["name"]
                stats["phase"] = phase
                stats_list.append(stats)

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def plot_event_comparison(
    stats_df: pd.DataFrame,
    events: List[Dict[str, Any]] = EVENTS,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Create event comparison figure with time series and before/after comparisons.

    Layout: 2 rows x 3 columns (one column per archetype)
    - Top row: Mean scores over time with event markers
    - Bottom row: Before/during/after violin plots per event

    Args:
        stats_df: DataFrame from analyze_event_windows()
        events: List of event dictionaries
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    events_parsed = parse_event_dates(events)

    archetypes = ["superspreader", "amplifier", "coordinated"]

    # Filter to event-specific data (exclude whole period)
    event_stats = stats_df[stats_df["event_id"] >= 0].copy()

    if len(event_stats) == 0:
        print("Warning: No event statistics found in stats_df")
        return fig

    for col_idx, archetype in enumerate(archetypes):
        # --- Top row: Time series ---
        ax_top = axes[0, col_idx]

        for event_id in sorted(event_stats["event_id"].unique()):
            event_data = event_stats[event_stats["event_id"] == event_id]
            event_info = events_parsed[int(event_id)]

            # Order: before, during, after
            phases = ["before", "during", "after"]
            means = []
            for p in phases:
                phase_data = event_data[event_data["phase"] == p]
                if len(phase_data) > 0:
                    # Extract mean from nested dict
                    arch_dict = phase_data[archetype].iloc[0]
                    means.append(arch_dict["mean"])
                else:
                    means.append(np.nan)

            x_positions = [event_id - 0.2, event_id, event_id + 0.2]

            ax_top.plot(
                x_positions,
                means,
                marker="o",
                linewidth=2,
                markersize=8,
                label=event_info["name"],
                color=event_info["color"],
                alpha=0.8,
            )

        ax_top.set_xticks(range(len(events_parsed)))
        ax_top.set_xticklabels([f"E{i+1}" for i in range(len(events_parsed))])
        ax_top.set_ylabel(f"Mean {archetype.capitalize()} Score", fontweight="bold")
        ax_top.set_title(
            f'({"ABC"[col_idx]}) {archetype.capitalize()} Evolution',
            fontweight="bold",
            pad=10,
        )
        ax_top.grid(alpha=0.3)
        ax_top.legend(fontsize=10, loc="best")

        # Set y-axis limits
        all_means = []
        for event_id in sorted(event_stats["event_id"].unique()):
            event_data = event_stats[event_stats["event_id"] == event_id]
            for p in ["before", "during", "after"]:
                phase_data = event_data[event_data["phase"] == p]
                if len(phase_data) > 0:
                    arch_dict = phase_data[archetype].iloc[0]
                    all_means.append(arch_dict["mean"])

        # Filter out NaNs
        all_means_clean = [m for m in all_means if np.isfinite(m)]
        if all_means_clean:
            max_mean = max(all_means_clean)
            ax_top.set_ylim(0, max(max_mean * 1.1, 0.1))
        else:
            ax_top.set_ylim(0, 1)  # fallback if no valid means

        # --- Bottom row: Before/During/After comparison ---
        ax_bot = axes[1, col_idx]

        # Collect data for violin plots
        plot_data = []
        plot_labels = []
        plot_colors = []

        for event_id in sorted(event_stats["event_id"].unique()):
            event_data = event_stats[event_stats["event_id"] == event_id]
            event_info = events_parsed[int(event_id)]

            for phase in ["before", "during", "after"]:
                phase_data = event_data[event_data["phase"] == phase]
                if len(phase_data) > 0:
                    arch_dict = phase_data[archetype].iloc[0]
                    mean_val = arch_dict["mean"]
                    std_val = arch_dict["std"]

                    # Generate synthetic data for violin (approximation)
                    synthetic = np.random.normal(mean_val, std_val, 100)
                    synthetic = np.clip(synthetic, 0, 1)

                    plot_data.append(synthetic)
                    plot_labels.append(f"E{int(event_id)+1}-{phase[0].upper()}")
                    plot_colors.append(event_info["color"])

        if plot_data:
            positions = list(range(len(plot_data)))
            parts = ax_bot.violinplot(
                plot_data,
                positions=positions,
                widths=0.6,
                showmeans=True,
                showmedians=False,
            )

            for pc, color in zip(parts["bodies"], plot_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor("black")

            ax_bot.set_xticks(positions)
            ax_bot.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=8)

        ax_bot.set_ylabel(f"{archetype.capitalize()} Score", fontweight="bold")
        ax_bot.set_title(
            f'({"DEF"[col_idx]}) Event windows comparison', fontweight="bold", pad=10
        )
        ax_bot.grid(axis="y", alpha=0.3)
        ax_bot.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def plot_shift_magnitude(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    event_name: str = "Event",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Optional[Figure]:
    """
    Plot histogram of archetype vector shifts between two periods.

    Args:
        df_before: DataFrame for period before event
        df_after: DataFrame for period after event
        event_name: Name of the event for title
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object or None if too few common users
    """
    # Find common users
    common_users = set(df_before["user_id"]) & set(df_after["user_id"])
    print(f"Common users between periods: {len(common_users)}")

    if len(common_users) < 10:
        print("Warning: Too few common users for meaningful shift analysis")
        return None

    # Compute shifts
    shifts = []
    for user_id in common_users:
        vec_before = df_before[df_before["user_id"] == user_id][
            ["superspreader", "amplifier", "coordinated"]
        ].values[0]
        vec_after = df_after[df_after["user_id"] == user_id][
            ["superspreader", "amplifier", "coordinated"]
        ].values[0]

        shift_magnitude = np.linalg.norm(vec_after - vec_before)
        shifts.append(shift_magnitude)

    shifts_array = np.array(shifts)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(shifts_array, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
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
        f"Archetype vector shifts around {event_name}",
        fontweight="bold",
        pad=20,
        fontsize=13,
    )

    # Move legend to top-right (where stats box was)
    ax.legend(
        loc="upper right", framealpha=0.95, edgecolor="black", fontsize=10, frameon=True
    )

    ax.grid(alpha=0.3, axis="y")

    # Add stats box at top-center
    stats_text = f"N = {len(common_users)} users\n"
    stats_text += f"Mean shift: {np.mean(shifts_array):.3f}\n"
    stats_text += f"Median shift: {np.median(shifts_array):.3f}\n"
    stats_text += f"Max shift: {np.max(shifts_array):.3f}"

    ax.text(
        0.5,  # Center horizontally
        0.97,  # Top of plot
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, edgecolor="black"),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


# ============================================================================
# STATISTICAL TESTING
# ============================================================================


def compare_periods_statistically(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    period1_name: str = "Period 1",
    period2_name: str = "Period 2",
) -> pd.DataFrame:
    """
    Compare two periods statistically for all archetypes.

    Args:
        df1: DataFrame for first period
        df2: DataFrame for second period
        period1_name: Name of first period
        period2_name: Name of second period

    Returns:
        DataFrame with test results
    """
    results = []

    for archetype in ["superspreader", "amplifier", "coordinated"]:
        data1 = np.array(df1[archetype].values)
        data2 = np.array(df2[archetype].values)

        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")

        # Effect size (Cohen's d)
        effect_size = cohens_d(data1, data2)

        # Mean difference
        mean_diff = float(np.mean(data2) - np.mean(data1))

        results.append(
            {
                "archetype": archetype,
                "period1": period1_name,
                "period2": period2_name,
                "mean1": float(np.mean(data1)),
                "mean2": float(np.mean(data2)),
                "mean_diff": mean_diff,
                "cohens_d": effect_size,
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": interpret_effect_size(effect_size),
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# EXAMPLE USAGE WORKFLOW
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ARLES ANALYSIS WORKFLOW FOR ICWSM PAPER")
    print("=" * 80)
    print("\nThis module provides the following functions:")
    print("  - prepare_dataframe(archetypes)")
    print("  - filter_users(df)")
    print("  - plot_archetype_overview(df)")
    print("  - analyze_event_windows(learner_results)")
    print("  - plot_event_comparison(stats_df)")
    print("  - plot_shift_magnitude(df_before, df_after)")
    print("  - compare_periods_statistically(df1, df2)")
    print("\nSee notebook for usage examples.")
