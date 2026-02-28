"""Shared constants, data-loading helpers, and styling for final paper outputs.

All scripts in this package import from here to keep configuration DRY.
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths (relative to repo root) ────────────────────────────────────
RESULTS_CSV = Path("data/2026_01_results_final.csv")
DATASET_CSV = Path("data/git_good_bench_merge_commits_all.csv")
FAIL_ONLY_AGGREGATE_CSV = Path("results/rq3_fail_only/aggregate_combined.csv")
FAIL_ONLY_PAIRED_CSV = Path("results/rq3_fail_only/paired_data.csv")
OUTPUT_DIR = Path("results/final_paper_figs")

# ── Model constants ──────────────────────────────────────────────────
MODELS_FULL = [
    "openai/gpt-5-nano",
    "groq:qwen/qwen3-32b",
    "local:meta-llama/Llama-3.1-8B-Instruct",
]

MODEL_SHORT = {
    "openai/gpt-5-nano": "GPT-5-nano",
    "groq:qwen/qwen3-32b": "Qwen3-32B",
    "local:meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
}

MODEL_ORDER = ["GPT-5-nano", "Qwen3-32B", "LLaMA-3.1-8B"]

# Short labels for Figure B axes (full names in caption)
MODEL_FIGB_LABELS = {
    "GPT-5-nano": "GPT-5n",
    "Qwen3-32B": "Qwen3-32B",
    "LLaMA-3.1-8B": "Llama-3.1-8B",
}

# For Table IV display
MODEL_TABLE_IV_NAMES = {
    "GPT-5-nano": "GPT-5-Nano",
    "Qwen3-32B": "Qwen3-32B",
    "LLaMA-3.1-8B": "Llama-3.1-8B-Instruct",
}

# For Table III display
FAIL_SOURCE_TO_MODEL = {
    "2025-11-09-gpt5nano-failure-classifications": "GPT-5 Nano",
    "2025-11-23-qwen3-failure-classifications": "Qwen3",
    "2026-02-03-llama-fail-classifications": "Llama",
}

MODEL_FULL_TO_TABLE3 = {
    "openai/gpt-5-nano": "GPT-5 Nano",
    "groq:qwen/qwen3-32b": "Qwen3",
    "local:meta-llama/Llama-3.1-8B-Instruct": "Llama",
}

# ── Method / eval constants ──────────────────────────────────────────
METHOD_ORDER = ["Base A", "Base B", "Agent", "Bypass"]
EVAL_MAP = {
    "base_a": "Base A",
    "base_b": "Base B",
    "agent": "Agent",
    "bypass7": "Bypass",
}
METHOD_COLORS = {
    "Base A": "#a6cee3",
    "Base B": "#b2df8a",
    "Agent": "#ff7f00",
    "Bypass": "#1f78b4",
}

# ── Performance metrics ──────────────────────────────────────────────
PERF = ["exact_match", "similarity", "bleu3", "rouge_l"]
PERF_LABELS = {
    "exact_match": "Exact Match",
    "similarity": "Similarity",
    "bleu3": "BLEU-3",
    "rouge_l": "ROUGE-L",
}

# ── Difficulty / size / bucket constants ─────────────────────────────
DIFF_ORDER = ["easy", "medium", "hard"]
SIZE_ORDER = ["small", "medium", "large", "huge"]
CONFLICT_BUCKETS = [(1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 999, "11+")]

# ── Style ────────────────────────────────────────────────────────────
def apply_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette("colorblind")


# ── Helpers ──────────────────────────────────────────────────────────
def sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def coerce_em(s: pd.Series) -> pd.Series:
    """Coerce an exact-match column to float 0/1."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0.5).astype(float)
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "1.0", "yes"])
        .astype(float)
    )


def save_fig(fig, path: Path):
    """Save a figure as both PDF and PNG, then close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        fig.savefig(str(path).replace(path.suffix, ext))
    plt.close(fig)
    logger.info(f"  Saved {path.name}")


# ── Data loading ─────────────────────────────────────────────────────
def load_results(path: Path | None = None) -> pd.DataFrame:
    """Load the main results CSV and add short model / method columns."""
    path = path or RESULTS_CSV
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    if "exact_match" in df.columns:
        df["exact_match"] = coerce_em(df["exact_match"])
    df["model"] = df["model_name"].map(MODEL_SHORT)
    df["method"] = df["eval_method"].map(EVAL_MAP)
    return df


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the GitGoodBench dataset CSV with scenario metadata."""
    path = path or DATASET_CSV
    df = pd.read_csv(path)
    # The first column might be unnamed index
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "row_idx"})
    df["id"] = df["id"].astype(str)
    return df


def load_scenario(path: Path | None = None) -> pd.DataFrame:
    """Load scenario metadata (conflict counts, repo stats) from the dataset CSV.

    The dataset CSV has an unnamed first column that serves as the numeric
    scenario ID (matching the ``id`` column in the results CSV).
    """
    path = path or DATASET_CSV
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        # Use the unnamed first column as the scenario ID (matches results CSV)
        e = {"id": str(r["Unnamed: 0"]) if "Unnamed: 0" in df.columns else str(r.name)}
        if "scenario" in df.columns:
            try:
                sc = ast.literal_eval(str(r["scenario"]))
                e["n_conflict_files"] = sc.get(
                    "number_of_files_with_merge_conflict", 0
                )
                e["n_total_conflicts"] = sc.get(
                    "total_number_of_merge_conflicts", 0
                )
            except (ValueError, SyntaxError):
                e["n_conflict_files"] = 0
                e["n_total_conflicts"] = 0
        for src, dst in [
            ("commits", "repo_commits"),
            ("code_lines", "repo_code_lines"),
            ("contributors", "repo_contributors"),
        ]:
            if src in df.columns:
                try:
                    e[dst] = int(r[src])
                except Exception:
                    e[dst] = 0
        for c in ["difficulty", "project_size"]:
            if c in df.columns:
                e[c] = str(r[c])
        rows.append(e)
    return pd.DataFrame(rows)


def common_ids(df: pd.DataFrame) -> set:
    """Return IDs that have both agent and bypass7 results across all models."""
    ab = df[df["eval_method"].isin(["agent", "bypass7"])]
    per_model = {
        m: set(ab[ab["model_name"] == m]["id"].unique())
        for m in ab["model_name"].dropna().unique()
    }
    return set.intersection(*per_model.values()) if per_model else set()


def instance_agg(df: pd.DataFrame, common: set) -> pd.DataFrame:
    """Aggregate per-file rows to per-instance (min EM, mean others)."""
    df = df[df["id"].isin(common)].copy()
    agg = {}
    for m in PERF:
        if m in df.columns:
            agg[m] = "min" if m == "exact_match" else "mean"
    for c in ["difficulty", "project_size"]:
        if c in df.columns:
            agg[c] = "first"
    return df.groupby(
        ["id", "model_name", "eval_method", "model", "method"], as_index=False
    ).agg(agg)
