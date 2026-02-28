"""Logistic regression effect plots for RQ2.

Quantifies which features predict multi-agent wins using interpretable models.
Shows odds ratios / coefficients with confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RQ2Config, DEFAULT_CONFIG, CHARACTERISTIC_DISPLAY_NAMES, METRIC_DISPLAY_NAMES
from .data import prepare_improvement_data, create_buckets, prepare_regression_data


# Set consistent theme
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)


@dataclass
class LogisticModelResult:
    """Container for logistic regression results.

    Attributes
    ----------
    coefficients : pd.DataFrame
        Dataframe with feature, coef, odds_ratio, ci_low, ci_high, p_value
    model : object
        The fitted model (sklearn or statsmodels)
    accuracy : float
        Model accuracy on training data
    n_samples : int
        Number of samples used
    """

    coefficients: pd.DataFrame
    model: object
    accuracy: float
    n_samples: int


def fit_logistic_model(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
) -> Optional[LogisticModelResult]:
    """Fit a logistic regression model to predict multi-agent wins.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric for win determination
    config : RQ2Config
        Configuration

    Returns
    -------
    LogisticModelResult or None
        Fitted model results, or None if fitting fails
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import logit
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False

    # Prepare improvement data
    improvement_data = prepare_improvement_data(df, config)
    if improvement_data.n_pairs == 0:
        return None

    work = create_buckets(improvement_data.dataframe, config)
    reg_data = prepare_regression_data(work, metric, config)

    if reg_data.empty or len(reg_data) < 50:
        return None

    # Prepare features
    categorical_cols = ["difficulty", "project_size", "file_type"]
    numeric_cols = ["conflict_size", "tokens_context", "tokens_original"]

    # Filter to available columns
    cat_available = [c for c in categorical_cols if c in reg_data.columns]
    num_available = [c for c in numeric_cols if c in reg_data.columns]

    if not cat_available and not num_available:
        return None

    # Create design matrix
    X_parts = []
    feature_names = []

    # One-hot encode categorical variables
    for col in cat_available:
        dummies = pd.get_dummies(reg_data[col], prefix=col, drop_first=True)
        X_parts.append(dummies)
        feature_names.extend(dummies.columns.tolist())

    # Standardize numeric variables
    for col in num_available:
        vals = pd.to_numeric(reg_data[col], errors="coerce")
        # Standardize (z-score)
        mean_val = vals.mean()
        std_val = vals.std()
        if std_val > 0:
            standardized = (vals - mean_val) / std_val
        else:
            standardized = vals - mean_val
        X_parts.append(standardized.to_frame(name=col))
        feature_names.append(col)

    if not X_parts:
        return None

    X = pd.concat(X_parts, axis=1)
    y = reg_data["win"].astype(int)

    # Drop rows with NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X) < 50:
        return None

    # Fit model
    if HAS_STATSMODELS:
        # Use statsmodels for confidence intervals
        X_with_const = sm.add_constant(X)
        try:
            model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=100)
        except Exception:
            return None

        # Extract coefficients and CIs
        coef_df = pd.DataFrame({
            "feature": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "z_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        })

        # Compute confidence intervals
        ci = model.conf_int(alpha=1 - config.ci_level)
        coef_df["ci_low"] = ci[0].values
        coef_df["ci_high"] = ci[1].values

        # Compute odds ratios
        coef_df["odds_ratio"] = np.exp(coef_df["coef"])
        coef_df["or_ci_low"] = np.exp(coef_df["ci_low"])
        coef_df["or_ci_high"] = np.exp(coef_df["ci_high"])

        # Remove intercept for plotting
        coef_df = coef_df[coef_df["feature"] != "const"]

        # Compute accuracy
        y_pred = (model.predict(X_with_const) > 0.5).astype(int)
        accuracy = (y_pred == y).mean()

        return LogisticModelResult(
            coefficients=coef_df,
            model=model,
            accuracy=float(accuracy),
            n_samples=len(X),
        )

    else:
        # Fallback to sklearn (no confidence intervals)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None

        model = LogisticRegression(max_iter=1000, random_state=config.random_state)
        model.fit(X, y)

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coef": model.coef_[0],
            "odds_ratio": np.exp(model.coef_[0]),
        })

        accuracy = model.score(X, y)

        return LogisticModelResult(
            coefficients=coef_df,
            model=model,
            accuracy=float(accuracy),
            n_samples=len(X),
        )


def render_odds_ratio_plot(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render an odds ratio plot for logistic regression coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric for win determination
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Fit model
    result = fit_logistic_model(df, metric, config)

    if result is None:
        fig, ax = plt.subplots(figsize=config.figsize_regression)
        ax.text(
            0.5, 0.5,
            "Unable to fit logistic regression.\nNeed statsmodels installed and sufficient data.",
            ha="center", va="center", fontsize=12,
        )
        ax.axis("off")
        return fig

    coef_df = result.coefficients
    if coef_df.empty:
        fig, ax = plt.subplots(figsize=config.figsize_regression)
        ax.text(0.5, 0.5, "No coefficients to display", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Sort by odds ratio
    coef_df = coef_df.sort_values("odds_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=config.figsize_regression)

    y_positions = np.arange(len(coef_df))
    features = coef_df["feature"].tolist()
    odds_ratios = coef_df["odds_ratio"].to_numpy()

    # Check if we have CIs
    has_ci = "or_ci_low" in coef_df.columns and "or_ci_high" in coef_df.columns

    # Color by effect direction
    colors = [
        config.positive_color if or_val > 1 else config.negative_color if or_val < 1 else config.neutral_color
        for or_val in odds_ratios
    ]

    # Draw error bars if available
    if has_ci:
        ci_lows = coef_df["or_ci_low"].to_numpy()
        ci_highs = coef_df["or_ci_high"].to_numpy()

        for i, (or_val, ci_low, ci_high, color) in enumerate(zip(odds_ratios, ci_lows, ci_highs, colors)):
            ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)
            ax.scatter(or_val, i, color=color, s=80, zorder=3, edgecolors="white", linewidths=1)
    else:
        for i, (or_val, color) in enumerate(zip(odds_ratios, colors)):
            ax.scatter(or_val, i, color=color, s=80, zorder=3, edgecolors="white", linewidths=1)

    # Add vertical line at odds ratio = 1 (no effect)
    ax.axvline(1, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add significance markers
    if "p_value" in coef_df.columns:
        for i, (or_val, p_val) in enumerate(zip(odds_ratios, coef_df["p_value"])):
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                ax.annotate(
                    marker, xy=(or_val, i), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=12, fontweight="bold",
                )

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.set_xlabel("Odds Ratio (Multi-Agent Win)")
    ax.set_xscale("log")

    # Add region labels
    xlim = ax.get_xlim()
    ax.text(
        xlim[1], len(coef_df) - 0.5, "Favors\nMulti-Agent",
        ha="right", va="top", fontsize=9, color=config.positive_color, fontweight="bold",
    )
    ax.text(
        xlim[0], len(coef_df) - 0.5, "Favors\nSingle-Agent",
        ha="left", va="top", fontsize=9, color=config.negative_color, fontweight="bold",
    )

    # Add model info
    info_text = f"n={result.n_samples}, Accuracy={result.accuracy:.1%}"
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(
        f"RQ2: Features Predicting Multi-Agent Wins\n({METRIC_DISPLAY_NAMES.get(metric, metric)})",
        fontweight="bold",
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def render_coefficient_plot(
    df: pd.DataFrame,
    metric: str = "exact_match",
    config: RQ2Config = DEFAULT_CONFIG,
    *,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Render a coefficient plot (log-odds) for logistic regression.

    Alternative to odds ratio plot - shows raw coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Input results dataframe
    metric : str
        The metric for win determination
    config : RQ2Config
        Configuration
    output_path : Path, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    result = fit_logistic_model(df, metric, config)

    if result is None:
        fig, ax = plt.subplots(figsize=config.figsize_regression)
        ax.text(0.5, 0.5, "Unable to fit model", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    coef_df = result.coefficients
    if coef_df.empty:
        fig, ax = plt.subplots(figsize=config.figsize_regression)
        ax.text(0.5, 0.5, "No coefficients to display", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Sort by coefficient
    coef_df = coef_df.sort_values("coef", ascending=True)

    fig, ax = plt.subplots(figsize=config.figsize_regression)

    y_positions = np.arange(len(coef_df))
    features = coef_df["feature"].tolist()
    coefs = coef_df["coef"].to_numpy()

    colors = [
        config.positive_color if c > 0 else config.negative_color if c < 0 else config.neutral_color
        for c in coefs
    ]

    # Draw error bars if available
    if "ci_low" in coef_df.columns and "ci_high" in coef_df.columns:
        ci_lows = coef_df["ci_low"].to_numpy()
        ci_highs = coef_df["ci_high"].to_numpy()

        for i, (coef, ci_low, ci_high, color) in enumerate(zip(coefs, ci_lows, ci_highs, colors)):
            ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)
            ax.scatter(coef, i, color=color, s=80, zorder=3, edgecolors="white", linewidths=1)
    else:
        for i, (coef, color) in enumerate(zip(coefs, colors)):
            ax.scatter(coef, i, color=color, s=80, zorder=3, edgecolors="white", linewidths=1)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.set_xlabel("Coefficient (Log-Odds)")

    info_text = f"n={result.n_samples}, Accuracy={result.accuracy:.1%}"
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(
        f"RQ2: Logistic Regression Coefficients\n({METRIC_DISPLAY_NAMES.get(metric, metric)})",
        fontweight="bold",
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
