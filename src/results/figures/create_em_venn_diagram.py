from __future__ import annotations

"""Create a Venn diagram of exact-match overlaps across up to 3 groups.

Groups are levels of a categorical column (default: `eval_method`). For each
group, we consider the set of unique instance keys (built from `id` if present,
else `repo`+`file_name`) for which `exact_match` is True, and plot their
overlaps via a 2- or 3-set Venn diagram.

Usage (examples):
  - python -m src.results.figures.create_em_venn_diagram --results-csv data/2025_10_18_Final_Results.csv --output-path results/venn_exact_match.png --group-by model_name --em-distributions --em-prefix results/em_dist

  - Select specific groups (must be 2 or 3):
    --group-by eval_method --include-groups methodA methodB methodC

  - Or pick top-k groups by number of exact matches (k in {2,3}):
    --group-by model_name --top-k 3

  - Additionally, you can generate exact-match distribution plots (bars and a heatmap):
    --em-distributions --em-prefix results/em_dist

    By default this creates:
      * results/em_dist_by_model_name.png (EM rate per model, single bar)
      * results/em_dist_by_eval_method.png (EM rate per method, single bar)
      * results/em_dist_heatmap_method_by_model.png (EM rate heatmap: method × model)

    To show two bars per model (one bar per selected method), either specify the two
    methods explicitly or let the script auto-pick the top two by sample count:
      * Explicit:
        --em-distributions --compare-methods methodA methodB
      * Auto-pick top two:
        --em-distributions   (no compare-methods provided; two most frequent methods used)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro

from ..data_loader import load_results
from ..diagnostics import _build_key


def _load_df_flexible(path: Optional[Path]) -> pd.DataFrame:
    """Load results with normalization when possible; fall back to raw CSV.

    Some historical result files may omit columns expected by the normalized
    loader (e.g., 'processing_time_s'). In those cases, read the CSV directly
    and minimally coerce 'exact_match' to boolean.
    """
    try:
        data = load_results(path)
        return data.dataframe
    except Exception:
        if path is None:
            # No concrete path to fall back to; re-raise
            raise
        df = pd.read_csv(path)
        if "exact_match" in df.columns and df["exact_match"].dtype != bool:
            df["exact_match"] = (
                df["exact_match"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
            )
        return df


@dataclass
class Flags:
    results_csv: Optional[Path] = None
    output_path: Path = Path("results/venn_exact_match.png")
    show: bool = True

    # Which column defines the groups/sets (e.g., eval_method, model_name)
    group_by: str = "eval_method"

    # Either explicitly include 2 or 3 groups, or let the script pick top-k by EM count
    include_groups: Optional[list[str]] = None
    top_k: int = 3  # only used when include_groups is None; must be 2 or 3

    # Optional title
    title: Optional[str] = None

    # When enabled, also create 3 EM distribution plots:
    #  - Bar chart of exact-match rate by model_name (if present)
    #  - Bar chart of exact-match rate by eval_method (if present)
    #  - Heatmap of exact-match rate by (eval_method x model_name) when both present
    em_distributions: bool = False
    em_prefix: Path = Path("results/em_dist")
    # Optional: When set, also render a grouped bar chart with two bars per model,
    # one per selected method. If not provided and two or more methods exist, the
    # top two by sample count will be auto-selected.
    compare_methods: Optional[list[str]] = None


def _validate_groups(df: pd.DataFrame, group_by: str, include_groups: Optional[Iterable[str]], top_k: int) -> list[str]:
    if group_by not in df.columns:
        raise ValueError(f"group_by column not found: {group_by}")

    if include_groups is not None and len(include_groups) > 0:
        present = set(df[group_by].astype(str).unique())
        missing = [g for g in include_groups if str(g) not in present]
        if missing:
            raise ValueError(f"Requested groups not present in column '{group_by}': {missing}")
        if len(include_groups) not in (2, 3):
            raise ValueError("Venn diagram supports exactly 2 or 3 groups when explicitly specified")
        return [str(g) for g in include_groups]

    # Auto-pick top-k groups by number of exact matches
    if top_k not in (2, 3):
        raise ValueError("When auto-selecting groups, top_k must be 2 or 3 for a Venn diagram")

    counts = (
        df.loc[df.get("exact_match", False).astype(bool)]
        .groupby(group_by)["exact_match"]
        .size()
        .sort_values(ascending=False)
    )
    if counts.empty:
        raise ValueError("No exact_match=True rows found; cannot create Venn diagram")

    chosen = counts.head(top_k).index.astype(str).tolist()
    return chosen


def _sets_by_group(df: pd.DataFrame, group_by: str, groups: list[str]) -> list[set[str]]:
    """Return a list of sets of instance keys, one per group, for which exact_match=True."""
    key = _build_key(df)
    work = df.assign(_key=key)
    sets: list[set[str]] = []
    for g in groups:
        subset = work[(work[group_by].astype(str) == g) & (work.get("exact_match", False).astype(bool))]
        sets.append(set(subset["_key"].astype(str).unique()))
    return sets


def _plot_venn(sets: list[set[str]], labels: list[str], *, title: Optional[str], output_path: Path, show: bool) -> None:
    from matplotlib_venn import venn2, venn3

    n = len(sets)
    if n not in (2, 3):
        raise ValueError("Venn plot only supports 2 or 3 sets")

    plt.figure(figsize=(7, 6))
    if n == 2:
        venn2(subsets=sets, set_labels=labels)
    else:
        venn3(subsets=sets, set_labels=labels)

    if title:
        plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _em_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute exact-match rate per group for the given column.

    Returns a DataFrame with columns: [group_col, 'count', 'em_count', 'em_rate'] sorted by em_rate desc.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column not found: {group_col}")
    work = df[[group_col, "exact_match"]].copy()
    work = work.dropna(subset=[group_col])
    work["exact_match"] = work["exact_match"].astype(bool)
    grp = work.groupby(group_col)["exact_match"]
    out = (
        pd.DataFrame({
            "count": grp.size(),
            "em_count": grp.sum(),
        })
        .assign(em_rate=lambda x: (x["em_count"] / x["count"]).astype(float))
        .sort_values("em_rate", ascending=False)
        .reset_index()
    )
    return out


def _plot_em_bar(stats: pd.DataFrame, group_col: str, *, save_path: Path, title: str, show: bool) -> None:
    labels = stats[group_col].astype(str).tolist()
    values = stats["em_rate"].astype(float).tolist()
    n = len(labels)
    width = max(7, min(20, 0.5 * n + 2))
    height = 5
    import matplotlib.pyplot as plt  # local import to avoid polluting global state
    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(range(n), values, color="#4C78A8")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Exact-match rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_em_by_model_two_methods(df: pd.DataFrame, methods: list[str], *, save_path: Path, title: str, show: bool) -> None:
    """Grouped bars: two bars per model_name, one per selected method, showing EM rate.

    Expects columns: ['eval_method', 'model_name', 'exact_match'].
    Methods should contain exactly two method names present in df.
    """
    if not {"eval_method", "model_name", "exact_match"}.issubset(df.columns):
        return
    if len(methods) != 2:
        return

    work = df[["eval_method", "model_name", "exact_match"]].copy()
    work = work.dropna(subset=["eval_method", "model_name"])  # require both
    work["eval_method"] = work["eval_method"].astype(str)
    work["model_name"] = work["model_name"].astype(str)
    work = work[work["eval_method"].isin(methods)]
    if work.empty:
        return

    work["exact_match"] = work["exact_match"].astype(bool)

    # compute EM rate per (model_name, eval_method)
    grp = work.groupby(["model_name", "eval_method"])  # type: ignore[index]
    stats = (
        pd.DataFrame({
            "count": grp.size(),
            "em_count": grp["exact_match"].sum(),
        })
        .assign(em_rate=lambda x: (x["em_count"] / x["count"]).astype(float))
        .reset_index()
    )

    # Determine model order and ensure both methods appear per model (fill missing with 0)
    models = stats["model_name"].astype(str).unique().tolist()
    meth_a, meth_b = methods[0], methods[1]
    pivot = stats.pivot(index="model_name", columns="eval_method", values="em_rate").reindex(index=models)
    for m in [meth_a, meth_b]:
        if m not in pivot.columns:
            pivot[m] = 0.0
    pivot = pivot[[meth_a, meth_b]].fillna(0.0)

    import matplotlib.pyplot as plt
    import numpy as np
    fig_w = max(8, 0.6 * len(models) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    x = np.arange(len(models))
    width = 0.38
    ax.bar(x - width/2, pivot[meth_a].values, width, label=meth_a, color="#4C78A8")
    ax.bar(x + width/2, pivot[meth_b].values, width, label=meth_b, color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Exact-match rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(title="method", loc="upper left", bbox_to_anchor=(1.02, 1))

    # small labels above bars
    for xi, (a, b) in enumerate(zip(pivot[meth_a].values, pivot[meth_b].values)):
        ax.text(xi - width/2, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width/2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_em_heatmap(df: pd.DataFrame, *, save_path: Path, title: str, show: bool) -> None:
    # Build pivot of EM rate by (eval_method x model_name)
    work = df[["eval_method", "model_name", "exact_match"]].copy()
    work = work.dropna(subset=["eval_method", "model_name"])  # require both
    work["exact_match"] = work["exact_match"].astype(bool)
    # compute rate per (method, model)
    grp = work.groupby(["eval_method", "model_name"])  # type: ignore[index]
    stats = pd.DataFrame({
        "count": grp.size(),
        "em_count": grp["exact_match"].sum(),
    }).assign(em_rate=lambda x: (x["em_count"] / x["count"]).astype(float))
    pivot = stats["em_rate"].unstack("model_name").fillna(0.0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * pivot.shape[1] + 2), max(5, 0.5 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=25, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Exact-match rate", rotation=90)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(flags: Flags) -> None:
    df = _load_df_flexible(flags.results_csv)

    if "exact_match" not in df.columns:
        raise ValueError("Results file must include 'exact_match' column")

    groups = _validate_groups(df, flags.group_by, flags.include_groups, flags.top_k)
    sets = _sets_by_group(df, flags.group_by, groups)

    # Default title if not provided
    if flags.title:
        title = flags.title
    else:
        title = f"Exact Match Overlap by {flags.group_by}: " + ", ".join(groups)

    _plot_venn(sets, groups, title=title, output_path=flags.output_path, show=flags.show)

    # Optional EM distribution plots
    if flags.em_distributions:
        if "exact_match" not in df.columns:
            raise ValueError("Results file must include 'exact_match' column for EM distribution plots")

        # 1) By model_name
        if "model_name" in df.columns:
            stats_model = _em_rate_table(df, "model_name")
            _plot_em_bar(
                stats_model,
                "model_name",
                save_path=Path(f"{flags.em_prefix}_by_model_name.png"),
                title="Exact-match rate by model_name",
                show=flags.show,
            )

        # 2) By eval_method
        if "eval_method" in df.columns:
            stats_method = _em_rate_table(df, "eval_method")
            _plot_em_bar(
                stats_method,
                "eval_method",
                save_path=Path(f"{flags.em_prefix}_by_eval_method.png"),
                title="Exact-match rate by method",
                show=flags.show,
            )

        # 3) Heatmap: eval_method x model_name
        if {"eval_method", "model_name"}.issubset(df.columns):
            _plot_em_heatmap(
                df,
                save_path=Path(f"{flags.em_prefix}_heatmap_method_by_model.png"),
                title="Exact-match rate: method × model_name",
                show=flags.show,
            )

        # 4) Two bars per model: one bar per selected method (EM rate)
        if {"eval_method", "model_name"}.issubset(df.columns):
            chosen_methods: Optional[list[str]] = None
            if flags.compare_methods and len(flags.compare_methods) >= 2:
                # take first two
                chosen_methods = [str(flags.compare_methods[0]), str(flags.compare_methods[1])]
            else:
                # auto-pick top two methods by sample count
                counts = df.groupby("eval_method").size().sort_values(ascending=False)
                if counts.shape[0] >= 2:
                    chosen_methods = counts.head(2).index.astype(str).tolist()

            if chosen_methods and len(chosen_methods) == 2:
                _plot_em_by_model_two_methods(
                    df,
                    chosen_methods,
                    save_path=Path(f"{flags.em_prefix}_by_model_two_methods.png"),
                    title=f"Exact-match rate by model (methods: {chosen_methods[0]} vs {chosen_methods[1]})",
                    show=flags.show,
                )


if __name__ == "__main__":
    parsed = tyro.cli(Flags)
    main(parsed)


