from __future__ import annotations

"""Interactive Plotly Dash dashboard for metrics by model+method.

Launch:
    python -m src.results.figures.metric_plots_by_model_dash --input-csv data/2025_10_07_ALL_RESULTS.csv --port 8050

The app provides controls for difficulty and metric, and renders boxplots where
the x-axis is unique `model_name · eval_method` and color encodes `eval_method`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import argparse
import re

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

from src.config.eval_methods import DEFAULT_METHOD_ORDER


def _slugify(s: object) -> str:
    try:
        s2 = str(s).strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")
        s2 = re.sub(r"[^a-z0-9_.-]+", "_", s2)
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2 or "metrics"
    except Exception:
        return "metrics"


def _coerce_metric(series: pd.Series, column: str) -> pd.Series:
    if column == "exact_match":
        s = series
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)
        return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)
    return pd.to_numeric(series, errors="coerce")


def _build_combo_label(model_name: object, eval_method: object) -> str:
    model = str(model_name).strip() if pd.notna(model_name) else "unknown"
    method = str(eval_method).strip()
    if not model:
        model = "unknown"
    if not method:
        method = "unknown"
    return f"{model} · {method}"


def _order_methods(present_methods: list[str]) -> list[str]:
    return [m for m in DEFAULT_METHOD_ORDER if m in present_methods] + [
        m for m in present_methods if m not in DEFAULT_METHOD_ORDER
    ]


def _order_combos(df: pd.DataFrame) -> list[str]:
    desired = DEFAULT_METHOD_ORDER
    df = df.copy()
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"
    df["eval_method"] = df["eval_method"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    combos: list[str] = []
    for m in desired:
        present = df.loc[df["eval_method"] == m, "model_name"].dropna().astype(str).unique().tolist()
        present = sorted({x.strip() if x.strip() else "unknown" for x in present})
        for model in present:
            combos.append(_build_combo_label(model, m))
    # add unseen methods at the end
    for m in df["eval_method"].astype(str).unique().tolist():
        if m in desired:
            continue
        present = df.loc[df["eval_method"] == m, "model_name"].dropna().astype(str).unique().tolist()
        present = sorted({x.strip() if x.strip() else "unknown" for x in present})
        for model in present:
            lab = _build_combo_label(model, m)
            if lab not in combos:
                combos.append(lab)
    return combos


def build_figure(df: pd.DataFrame, metric: str, difficulty: Optional[str]) -> "px.Figure":
    has_difficulty = "difficulty" in df.columns
    if has_difficulty:
        df["difficulty_norm"] = df["difficulty"].astype(str).str.strip().str.lower()
    if difficulty and has_difficulty:
        sub = df.loc[df["difficulty_norm"] == difficulty]
    else:
        sub = df

    if sub.empty or metric not in sub.columns:
        return px.box(pd.DataFrame({"x": [], metric: []}), x="x", y=metric, title="No data")

    work = pd.DataFrame({
        "model_method": [_build_combo_label(m, e) for m, e in zip(sub.get("model_name", "unknown"), sub["eval_method"])],
        "eval_method": sub["eval_method"].astype(str),
        metric: _coerce_metric(sub[metric], metric),
    })
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    if metric == "exact_match":
        work = work.groupby(["model_method", "eval_method"], as_index=False)[metric].mean()

    method_order = _order_methods(work["eval_method"].unique().tolist())
    combo_order = _order_combos(work.rename(columns={"model_method": "model_method_tmp"}).assign(model_name=work["model_method"].str.split(" · ").str[0], eval_method=work["model_method"].str.split(" · ").str[1]))

    fig = px.box(
        work,
        x="model_method",
        y=metric,
        color="eval_method",
        category_orders={"eval_method": method_order, "model_method": combo_order},
        points=False,
    )
    fig.update_layout(
        xaxis_title="model · method",
        yaxis_title=metric,
        boxmode="group",
        legend_title_text="method",
        margin=dict(l=20, r=20, t=50, b=80),
        height=600,
    )
    fig.update_xaxes(tickangle=30)
    if metric in {"exact_match", "similarity", "bleu3", "rouge_l"}:
        fig.update_yaxes(range=[0, 1])
    return fig


@dataclass
class Flags:
    input_csv: Path
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False


def main(flags: Flags) -> None:
    df = pd.read_csv(flags.input_csv)
    if "eval_method" not in df.columns:
        raise ValueError("Column 'eval_method' not found in input CSV; required for comparisons.")
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"

    app = Dash(__name__)

    metrics = [m for m in ["exact_match", "similarity", "bleu3", "rouge_l"] if m in df.columns]
    has_difficulty = "difficulty" in df.columns
    difficulties = ["all"]
    if has_difficulty:
        difficulties += sorted(df["difficulty"].astype(str).str.strip().str.lower().unique().tolist())

    app.layout = html.Div(
        children=[
            html.H3("Metrics by Model · Method"),
            html.Div(
                children=[
                    html.Label("Metric"),
                    dcc.Dropdown(id="metric", options=[{"label": m, "value": m} for m in metrics], value=metrics[0], clearable=False, style={"width": "240px"}),
                    html.Label("Difficulty", style={"marginLeft": "16px"}),
                    dcc.Dropdown(id="difficulty", options=[{"label": d.title(), "value": (None if d == "all" else d)} for d in difficulties], value=None, clearable=False, style={"width": "200px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "10px"},
            ),
            dcc.Graph(id="fig"),
        ],
        style={"padding": "12px"},
    )

    @app.callback(Output("fig", "figure"), Input("metric", "value"), Input("difficulty", "value"))
    def _update(metric: str, difficulty: Optional[str]):  # type: ignore[override]
        return build_figure(df, metric, difficulty)

    app.run(host=flags.host, port=flags.port, debug=flags.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(Flags(input_csv=args.input_csv, host=args.host, port=args.port, debug=args.debug))


