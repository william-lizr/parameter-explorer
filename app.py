"""No-code parameter-sweep explorer.

Load a tidy CSV (one row per simulation run). Mark each column as a
parameter, a metric, or a result (a timeseries or matrix stored in the cell).
Drag parameters between "unconstrained" and "constrained" to slice the sweep.
The right side shows the analytics you pick for the rows that remain.

Run:  python app.py [optional.csv]
Production:  gunicorn app:server  (see Dockerfile)
"""
import base64
import io
import json
import os
import re
import sys
import uuid
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import mpl_export

# ─── APP ──────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Parameter Explorer",
)
server = app.server

# Datasets live on the server. The browser only holds a key. This keeps big
# result columns (timeseries, matrices) out of the browser.
# On a shared server, keep only the newest MAX_DATASETS so memory stays bounded.
DATASETS: dict[str, pd.DataFrame] = {}
MAX_DATASETS = int(os.environ.get("MAX_DATASETS", "20"))

# The CSV format guide. The same file is the GitHub wiki page.
HELP_FILE = Path(__file__).parent / "docs" / "CSV-format.md"
HELP_MD = HELP_FILE.read_text(encoding="utf-8") if HELP_FILE.is_file() else "Guide not found."

# Example CSVs made by generate_sample.py.
SAMPLES_DIR = Path(__file__).parent / "samples"


def list_samples():
    return sorted(p.name for p in SAMPLES_DIR.glob("*.csv")) if SAMPLES_DIR.is_dir() else []

# ─── STYLES ───────────────────────────────────────────────────────────────────

ACCENT = "#17a2b8"
BG = "#111318"

SECTION_LABEL_STYLE = {
    "fontSize": "10px",
    "fontWeight": "600",
    "letterSpacing": "1.5px",
    "color": "#9aa3b2",
    "marginBottom": "4px",
    "marginTop": "6px",
}

BADGE_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "marginBottom": "4px",
    "padding": "5px 8px",
    "fontSize": "12px",
    "width": "100%",
    "borderRadius": "4px",
    "backgroundColor": "#2c3140",
    "border": "1px solid #3a4255",
}

AXIS_DARK = dict(gridcolor="#2a2a2a", zerolinecolor="#333")
SCENE_AXIS_DARK = dict(gridcolor="#2a2a2a", zerolinecolor="#333", backgroundcolor=BG)

ANALYTICS = [
    {"label": "Auto (by number of varying params)", "value": "auto"},
    {"label": "3D surface", "value": "3d"},
    {"label": "3D point map (3 varying params)", "value": "points"},
    {"label": "Split by fewest-valued param", "value": "split"},
    {"label": "Heatmap", "value": "heatmap"},
    {"label": "Slices (metric vs param)", "value": "2d"},
    {"label": "Sensitivity (each varying param)", "value": "sensitivity"},
    {"label": "Metric pairs", "value": "pairs"},
    {"label": "Distribution", "value": "hist"},
    {"label": "Result inspector", "value": "inspector"},
]
ANALYTICS_LABEL = {a["value"]: a["label"] for a in ANALYTICS}

# ─── HELPERS: DATA ────────────────────────────────────────────────────────────

def looks_like_result(sample: str) -> bool:
    """A cell holds a result if it is a long list of numbers (JSON or delimited)."""
    s = sample.strip()
    return s.startswith("[") or s.count(",") > 5 or s.count(";") > 5


def detect_col_types(df: pd.DataFrame) -> dict:
    col_types = {}
    n = len(df)
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            sample = str(non_null.iloc[0]) if len(non_null) > 0 else ""
            col_types[col] = "result" if looks_like_result(sample) else "parameter"
        else:
            n_unique = series.nunique()
            col_types[col] = "parameter" if (n_unique <= 50 or n_unique / max(n, 1) < 0.25) else "metric"
    return col_types


def register_df(df: pd.DataFrame) -> str:
    key = uuid.uuid4().hex
    DATASETS[key] = df.reset_index(drop=True)
    while len(DATASETS) > MAX_DATASETS:  # dicts keep insertion order: drop the oldest
        DATASETS.pop(next(iter(DATASETS)))
    return key


def get_df(key):
    return DATASETS.get(key) if key else None


def apply_constraints(df: pd.DataFrame, constrained: dict) -> pd.DataFrame:
    filtered = df
    for col, spec in constrained.items():
        if col not in filtered.columns:
            continue
        if spec["type"] == "range":
            lo, hi = spec["range"]
            filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]
        elif spec["type"] == "values":
            filtered = filtered[filtered[col].isin(spec.get("values") or [])]
    return filtered


def default_constraint(vals: pd.Series) -> dict:
    if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 8:
        return {"type": "range", "range": [float(vals.min()), float(vals.max())]}
    return {"type": "values", "values": sorted(vals.unique().tolist())}


def parse_result(cell) -> np.ndarray | None:
    """Turn a result cell into a numeric array (1D timeseries or 2D matrix)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    s = str(cell).strip()
    try:
        arr = np.asarray(json.loads(s), dtype=float)
        return arr if arr.ndim in (1, 2) else arr.ravel()
    except (ValueError, TypeError):
        pass
    # Fallback: rows split by ";" or newlines, values by commas or spaces.
    rows = [r for r in re.split(r"[;\n]", s) if r.strip()]
    nums = [[float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", r)] for r in rows]
    nums = [r for r in nums if r]
    if not nums:
        return None
    if len(nums) > 1 and len(nums[0]) > 1 and len({len(r) for r in nums}) == 1:
        return np.asarray(nums)
    return np.asarray([x for r in nums for x in r])


def numeric_view(series: pd.Series) -> np.ndarray:
    """Numbers for plotting. Categorical columns become integer codes."""
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    return series.astype("category").cat.codes.to_numpy(dtype=float)


# ─── HELPERS: FIGURES ─────────────────────────────────────────────────────────

def style_fig(fig: go.Figure, uirev: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color="#ccc",
        margin=dict(l=50, r=20, t=30, b=45),
        uirevision=uirev,
    )
    fig.update_xaxes(**AXIS_DARK)
    fig.update_yaxes(**AXIS_DARK)
    auto_legend(fig)
    return fig


def auto_legend(fig: go.Figure) -> None:
    """Show a legend only when it tells the reader something: 2+ named traces.
    Few entries go in a row above the plot. Many entries go in a column on the right."""
    entries = {t.legendgroup or t.name for t in fig.data
               if t.name and t.showlegend is not False}
    if len(entries) < 2:
        fig.update_layout(showlegend=False)
        return
    if len(entries) <= 5:
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(t=40),
        )
    else:
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10), title=dict(text="")),
        )


def make_empty(msg: str = "Load a CSV to begin") -> go.Figure:
    fig = go.Figure(layout=go.Layout(
        annotations=[{
            "text": msg.replace("\n", "<br>"), "showarrow": False,
            "font": {"size": 13, "color": "#8a8f9c"},
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
        }],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    ))
    return style_fig(fig)


def grid_mean(df, p1, p2, metric):
    """Mean of the metric on the (p1, p2) grid, plus one row index per cell for clicks."""
    tmp = df[[p1, p2, metric]].copy()
    tmp["_row"] = df.index
    z = tmp.pivot_table(index=p2, columns=p1, values=metric, aggfunc="mean")
    rows = tmp.pivot_table(index=p2, columns=p1, values="_row", aggfunc="first")
    rows = rows.reindex(index=z.index, columns=z.columns)
    per_cell = len(tmp) / max(z.notna().sum().sum(), 1)
    return z, rows, per_cell


def averaged_note(dims, used, per_cell):
    rest = [p for p in dims if p not in used]
    parts = []
    if rest:
        parts.append(f"averaged over {', '.join(rest)}")
    elif per_cell > 1.01:
        parts.append(f"mean of ~{per_cell:.0f} rows per cell")
    return f" · {'; '.join(parts)}" if parts else ""


def fig_hist(df, metric):
    fig = go.Figure(go.Histogram(
        x=df[metric], nbinsx=40,
        marker=dict(color=ACCENT, line=dict(color="#0d6e7e", width=0.5)),
    ))
    fig.update_layout(xaxis_title=metric, yaxis_title="Count")
    return fig, f"Distribution of {metric}"


def fig_slices(df, dims, metric, colorby):
    """Metric vs the first free param. One line per value of the second free param."""
    if not dims:
        return fig_hist(df, metric)
    p1 = dims[0]
    hover = f"{p1}: %{{x:.4g}}<br>{metric}: %{{y:.4g}}<extra></extra>"

    if len(dims) == 1 and colorby and colorby in df.columns and colorby != metric:
        fig = go.Figure(go.Scatter(
            x=df[p1], y=df[metric], mode="markers", customdata=df.index,
            marker=dict(color=numeric_view(df[colorby]), colorscale="Viridis", showscale=True,
                        colorbar=dict(title=colorby), size=6, opacity=0.85),
            hovertemplate=hover,
        ))
        fig.update_layout(xaxis_title=p1, yaxis_title=metric)
        return fig, f"{metric} vs {p1} · color = {colorby}"

    fig = go.Figure()
    if len(dims) == 1:
        g = df.groupby(p1)[metric].agg(["mean", "std"]).reset_index()
        g["std"] = g["std"].fillna(0)
        x = g[p1]
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([g["mean"] + g["std"], (g["mean"] - g["std"])[::-1]]),
            fill="toself", fillcolor="rgba(23,162,184,0.15)", line=dict(width=0),
            hoverinfo="skip", name="± 1 std",
        ))
        fig.add_trace(go.Scatter(x=x, y=g["mean"], mode="lines", line=dict(color=ACCENT, width=2),
                                 hoverinfo="skip", name="mean"))
        fig.add_trace(go.Scatter(x=df[p1], y=df[metric], mode="markers", customdata=df.index,
                                 marker=dict(color=ACCENT, size=4, opacity=0.5),
                                 hovertemplate=hover, name="single runs"))
        fig.update_layout(xaxis_title=p1, yaxis_title=metric)
        return fig, f"{metric} vs {p1} (mean ± std)"

    p2 = dims[1]
    levels = sorted(df[p2].unique())
    if len(levels) > 12:
        # Too many lines to read. Pick 12 evenly spaced levels.
        levels = [levels[i] for i in np.linspace(0, len(levels) - 1, 12).round().astype(int)]
    colors = _sample_colorscale(len(levels))
    for lvl, col in zip(levels, colors):
        sub = df[df[p2] == lvl]
        g = sub.groupby(p1)[metric].mean()
        first_row = sub.reset_index().groupby(p1)["index"].first().reindex(g.index)
        fig.add_trace(go.Scatter(
            x=g.index, y=g.values, mode="lines+markers", name=f"{p2}={lvl:.4g}" if isinstance(lvl, (int, float, np.number)) else f"{p2}={lvl}",
            customdata=first_row.values,
            line=dict(color=col, width=1.5), marker=dict(size=4),
            hovertemplate=f"{p1}: %{{x:.4g}}<br>{metric}: %{{y:.4g}}<extra>{p2}={lvl}</extra>",
        ))
    fig.update_layout(xaxis_title=p1, yaxis_title=metric)
    note = averaged_note(dims, [p1, p2], 1)
    return fig, f"{metric} vs {p1}, one line per {p2}{note}"


def _sample_colorscale(n):
    from plotly.colors import sample_colorscale
    return sample_colorscale("Plasma", [i / max(n - 1, 1) * 0.9 for i in range(n)])


def fig_3d(df, dims, metric, colorby):
    if len(dims) < 2:
        return fig_slices(df, dims, metric, colorby)
    p1, p2 = dims[0], dims[1]
    z, rows, per_cell = grid_mean(df, p1, p2, metric)
    full_grid = z.notna().all().all() and z.shape[0] > 1 and z.shape[1] > 1

    if full_grid:
        fig = go.Figure(go.Surface(
            x=z.columns.values, y=z.index.values, z=z.values,
            customdata=rows.values,
            colorscale="Plasma", colorbar=dict(title=metric),
            hovertemplate=f"{p1}: %{{x:.4g}}<br>{p2}: %{{y:.4g}}<br>{metric}: %{{z:.4g}}<extra></extra>",
        ))
    else:
        c = numeric_view(df[colorby]) if colorby and colorby in df.columns else df[metric]
        fig = go.Figure(go.Scatter3d(
            x=df[p1], y=df[p2], z=df[metric], mode="markers", customdata=df.index,
            marker=dict(size=3, color=c, colorscale="Plasma", showscale=True,
                        colorbar=dict(title=colorby or metric), opacity=0.85),
            hovertemplate=f"{p1}: %{{x:.4g}}<br>{p2}: %{{y:.4g}}<br>{metric}: %{{z:.4g}}<extra></extra>",
        ))
    fig.update_layout(scene=dict(
        xaxis=dict(title=p1, **SCENE_AXIS_DARK),
        yaxis=dict(title=p2, **SCENE_AXIS_DARK),
        zaxis=dict(title=metric, **SCENE_AXIS_DARK),
        bgcolor=BG,
    ))
    kind = "surface" if full_grid else "scatter (not a full grid)"
    return fig, f"{metric} over {p1} × {p2} · {kind}{averaged_note(dims, [p1, p2], per_cell)}"


def fig_points(df, dims, metric, colorby, encoding="both"):
    """Points in the 3D space of three varying params. The metric sets point size, color, or both."""
    if len(dims) < 3:
        return make_empty(f"The point map needs 3 varying params. Now {len(dims)} vary: "
                          f"{', '.join(dims) or 'none'}."), "3D point map"
    x, y, z = dims[:3]

    # One point per (x, y, z) cell: mean over repeated runs, if any.
    color_col = colorby if colorby and colorby in df.columns and colorby not in (x, y, z) else None
    tmp = df[[x, y, z, metric]].copy()
    tmp["_row"] = df.index
    agg = {metric: "mean", "_row": "first"}
    if color_col and color_col != metric:
        tmp[color_col] = numeric_view(df[color_col])
        agg[color_col] = "mean"
    g = tmp.groupby([x, y, z], sort=False).agg(agg).reset_index()
    per_point = len(tmp) / max(len(g), 1)

    val = g[metric].to_numpy(dtype=float)
    lo, hi = (np.nanmin(val), np.nanmax(val)) if np.isfinite(val).any() else (0.0, 1.0)

    def size_of(v):
        # Area grows with the metric: 4 px across at the minimum, 24 px at the maximum.
        v = np.asarray(v, dtype=float)
        frac = (v - lo) / (hi - lo) if hi > lo else np.full(v.shape, 0.5)
        return 4 + 20 * np.sqrt(np.clip(np.nan_to_num(frac, nan=0.0), 0, 1))

    use_size = encoding in ("both", "size")
    use_color = encoding in ("both", "color") or color_col is not None
    c_name = color_col or metric

    marker = dict(size=size_of(val) if use_size else 6, sizemode="diameter",
                  opacity=0.85, line=dict(width=0))
    if use_color:
        marker.update(color=g[c_name].to_numpy(dtype=float), colorscale="Plasma",
                      showscale=True, colorbar=dict(title=c_name))
    else:
        marker.update(color=ACCENT)

    fig = go.Figure(go.Scatter3d(
        x=g[x], y=g[y], z=g[z], mode="markers", customdata=g["_row"],
        text=[f"{v:.4g}" for v in val], marker=marker, showlegend=False,
        hovertemplate=(f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<br>{z}: %{{z:.4g}}"
                       f"<br>{metric}: %{{text}}<extra></extra>"),
    ))

    # Size legend: three reference points that only appear in the legend.
    if use_size and hi > lo:
        for v in (lo, (lo + hi) / 2, hi):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers", name=f"{metric} = {v:.3g}",
                marker=dict(size=float(size_of([v])[0]), color="#b4b9c4", line=dict(width=0)),
                hoverinfo="skip",
            ))

    fig.update_layout(scene=dict(
        xaxis=dict(title=x, **SCENE_AXIS_DARK),
        yaxis=dict(title=y, **SCENE_AXIS_DARK),
        zaxis=dict(title=z, **SCENE_AXIS_DARK),
        bgcolor=BG,
    ))
    enc = {"both": "size + color", "size": "size", "color": "color"}[encoding]
    note = f" · mean of ~{per_point:.0f} rows per point" if per_point > 1.01 else ""
    return fig, f"{metric} over {x} × {y} × {z} · {enc}{note}"


def fig_split(df, varying, metric):
    """Small multiples: one 2D map per value of the param with the fewest values.
    The other two varying params are the axes. The metric is the color, on one
    shared scale so the panels compare directly."""
    if len(varying) < 3:
        return make_empty(f"Split view needs 3 varying params. Now {len(varying)} vary: "
                          f"{', '.join(varying) or 'none'}."), "Split by param"
    split = split_param(varying)
    x, y = [p for p in varying if p != split][:2]
    levels = sorted(df[split].dropna().unique())

    n = len(levels)
    cols = n if n <= 3 else int(np.ceil(n / 2))
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{split} = {_fmt(v)}" for v in levels],
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.04, vertical_spacing=0.1 if rows > 1 else 0.05)

    # One colour scale for every panel.
    zmin, zmax = float(df[metric].min()), float(df[metric].max())
    best = None  # (value, level index, x, y) of the best cell across all panels
    per_cell = 1.0
    for k, lvl in enumerate(levels):
        r, c = k // cols + 1, k % cols + 1
        sub = df[df[split] == lvl]
        z, rws, per_cell = grid_mean(sub, x, y, metric)
        sparse = z.isna().to_numpy().mean() > 0.5
        hover = (f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<br>{metric}: %{{z:.4g}}"
                 f"<extra>{split} = {_fmt(lvl)}</extra>")
        if not sparse:
            fig.add_trace(go.Heatmap(
                x=z.columns.values, y=z.index.values, z=z.values, customdata=rws.values,
                zmin=zmin, zmax=zmax, colorscale="Plasma", showscale=(k == 0),
                colorbar=dict(title=metric), hovertemplate=hover,
            ), row=r, col=c)
            if z.notna().any().any():
                iy, ix = np.unravel_index(np.nanargmax(z.values), z.shape)
                cand = (float(z.values[iy, ix]), k, z.columns.values[ix], z.index.values[iy])
                best = cand if best is None or cand[0] > best[0] else best
        else:
            # Not a grid (for example random search): plain coloured points.
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers", customdata=sub.index,
                marker=dict(color=sub[metric], cmin=zmin, cmax=zmax, colorscale="Plasma",
                            showscale=(k == 0), colorbar=dict(title=metric), size=6),
                hovertemplate=hover.replace("%{z:.4g}", "%{marker.color:.4g}"), showlegend=False,
            ), row=r, col=c)
        if r == rows or k + cols >= n:
            fig.update_xaxes(title_text=x, row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text=y, row=r, col=c)

    if best is not None:
        _, k, bx, by = best
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode="markers", name=f"max {metric}",
            marker=dict(symbol="x", size=12, color="white"), hoverinfo="skip",
        ), row=k // cols + 1, col=k % cols + 1)
    fig.update_annotations(font_size=11)

    note = f" · mean of ~{per_cell:.0f} rows per cell" if per_cell > 1.01 else ""
    return fig, f"{metric} over {x} × {y}, split by {split} ({varying[split]} values){note}"


def fig_heatmap(df, dims, metric, colorby):
    if len(dims) < 2:
        return fig_slices(df, dims, metric, colorby)
    p1, p2 = dims[0], dims[1]
    z, rows, per_cell = grid_mean(df, p1, p2, metric)
    fig = go.Figure(go.Heatmap(
        x=z.columns.values, y=z.index.values, z=z.values, customdata=rows.values,
        name=f"mean {metric}", showlegend=True,
        colorscale="Plasma", colorbar=dict(title=metric),
        hovertemplate=f"{p1}: %{{x:.4g}}<br>{p2}: %{{y:.4g}}<br>{metric}: %{{z:.4g}}<extra></extra>",
    ))
    # Mark the best cell so it is easy to find.
    if z.notna().any().any():
        iy, ix = np.unravel_index(np.nanargmax(z.values), z.shape)
        fig.add_trace(go.Scatter(
            x=[z.columns.values[ix]], y=[z.index.values[iy]], mode="markers",
            marker=dict(symbol="x", size=12, color="white"), hoverinfo="skip", name=f"max {metric}",
        ))
    fig.update_layout(xaxis_title=p1, yaxis_title=p2)
    return fig, f"{metric} over {p1} × {p2} · × = max{averaged_note(dims, [p1, p2], per_cell)}"


def eta_squared(x: pd.Series, y: pd.Series) -> float:
    """Share of the metric's variance explained by this param alone (0 to 1)."""
    total = ((y - y.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    g = y.groupby(x.values)
    between = (g.count() * (g.mean() - y.mean()) ** 2).sum()
    return float(between / total)


def fig_sensitivity(df, dims, metrics):
    """Grid: one row per metric, one column per free param. Mean ± std of the
    metric at each param value, taken over all other free params."""
    if not dims:
        return make_empty("No varying parameters. Release a constraint."), "Sensitivity"
    metrics = metrics[:4]
    params = dims[:6]

    # Bin params with many distinct values so the groups have several rows.
    groups = {}
    for p in params:
        if pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 30:
            groups[p] = pd.cut(df[p], 20).apply(lambda iv: iv.mid).astype(float)
        else:
            groups[p] = df[p]

    titles = []
    for m in metrics:
        for p in params:
            titles.append(f"{p} · η²={eta_squared(groups[p], df[m]):.2f}")

    fig = make_subplots(rows=len(metrics), cols=len(params), subplot_titles=titles,
                        shared_yaxes="rows", horizontal_spacing=0.03, vertical_spacing=0.12)
    for i, m in enumerate(metrics, start=1):
        for j, p in enumerate(params, start=1):
            g = df[m].groupby(groups[p].values).agg(["mean", "std"])
            g["std"] = g["std"].fillna(0)
            x = g.index.to_numpy()
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([(g["mean"] + g["std"]).values, (g["mean"] - g["std"]).values[::-1]]),
                fill="toself", fillcolor="rgba(23,162,184,0.15)", line=dict(width=0),
                hoverinfo="skip", name="± 1 std", legendgroup="std", showlegend=(i == j == 1),
            ), row=i, col=j)
            fig.add_trace(go.Scatter(
                x=x, y=g["mean"].values, mode="lines+markers",
                line=dict(color=ACCENT, width=2), marker=dict(size=4),
                name="mean over other params", legendgroup="mean", showlegend=(i == j == 1),
                hovertemplate=f"{p}: %{{x:.4g}}<br>mean {m}: %{{y:.4g}}<extra></extra>",
            ), row=i, col=j)
            if j == 1:
                fig.update_yaxes(title_text=m, row=i, col=j)
    fig.update_annotations(font_size=10)
    note = f" · first {len(params)} params" if len(dims) > len(params) else ""
    return fig, f"Sensitivity: η² = variance share explained by each param alone{note}"


def fig_pairs(df, metrics, colorby):
    if len(metrics) < 2:
        return make_empty("Select 2 or more metrics"), "Metric pairs"
    metrics = metrics[:6]
    c_col = colorby if colorby and colorby in df.columns else metrics[0]
    fig = go.Figure(go.Splom(
        dimensions=[dict(label=m, values=df[m]) for m in metrics],
        customdata=df.index,
        marker=dict(size=3, color=numeric_view(df[c_col]), colorscale="Plasma",
                    showscale=True, colorbar=dict(title=c_col), opacity=0.7),
        diagonal_visible=False, showupperhalf=False,
    ))
    fig.update_layout(dragmode="select")
    return fig, f"Metric pairs · color = {c_col}"


def fig_inspector(df_full, row, col_types, param_cols, metrics):
    result_cols = [c for c, t in col_types.items() if t == "result" and c in df_full.columns][:4]
    if not result_cols:
        return make_empty("No result columns. Mark a column as 'result' under Column types."), "Result inspector"
    if row is None or row not in df_full.index:
        return make_empty("Click a point in any plot to inspect its results"), "Result inspector"

    rec = df_full.loc[row]
    label = ", ".join(f"{p}={rec[p]:.4g}" if isinstance(rec[p], (int, float, np.number)) else f"{p}={rec[p]}"
                      for p in param_cols)
    fig = make_subplots(rows=1, cols=len(result_cols), subplot_titles=result_cols,
                        horizontal_spacing=0.08)
    for j, c in enumerate(result_cols, start=1):
        arr = parse_result(rec[c])
        if arr is None:
            continue
        if arr.ndim == 2:
            fig.add_trace(go.Heatmap(
                z=arr, colorscale="RdBu_r", zmid=0, showscale=(j == len(result_cols)),
                hovertemplate="row %{y}, col %{x}: %{z:.3g}<extra></extra>",
            ), row=1, col=j)
            # Square cells, and no empty padding around the matrix.
            fig.update_xaxes(constrain="domain", row=1, col=j)
            fig.update_yaxes(autorange="reversed", scaleanchor=f"x{j if j > 1 else ''}",
                             constrain="domain", row=1, col=j)
        else:
            fig.add_trace(go.Scatter(y=arr, mode="lines", line=dict(color=ACCENT, width=1.2),
                                     showlegend=False), row=1, col=j)
    fig.update_annotations(font_size=11)
    return fig, f"Result inspector · row {row} · {label}"


# The split view needs a param with this many values or fewer.
SPLIT_MAX = 5


def varying_params(df, param_cols, free_params):
    """Params that still take more than one value in the filtered rows, as
    {name: number of values}. Free params come first, then constrained params
    that kept several values."""
    ordered = list(free_params) + [p for p in param_cols if p not in free_params]
    counts = {p: int(df[p].nunique()) for p in ordered if p in df.columns}
    return {p: n for p, n in counts.items() if n > 1}


def split_param(varying):
    """The varying param with the fewest values. Ties go to the later column,
    so the first params stay on the axes."""
    if not varying:
        return None
    return min(reversed(list(varying)), key=lambda p: varying[p])


def resolve_auto(view, varying):
    if view != "auto":
        return view
    n = len(varying)
    if n == 3:
        return "split" if varying[split_param(varying)] <= SPLIT_MAX else "points"
    return {0: "hist", 1: "2d", 2: "3d"}.get(n, "blocked")


def varying_text(varying):
    return ", ".join(f"{p} ({n})" for p, n in varying.items()) or "none"


def blocked_reason(view, varying):
    """Hard stop. A plot may show at most 2 varying params. There are two
    exceptions with exactly 3: the 3D point map, and the split view (one 2D
    plot per value of a param with few values). Returns a message, or None if allowed."""
    n = len(varying)
    if view == "points":
        if n == 3:
            return None
        return f"The 3D point map needs exactly 3 varying params.\nNow {n} vary: {varying_text(varying)}."
    if view == "split":
        if n != 3:
            return f"The split view needs exactly 3 varying params.\nNow {n} vary: {varying_text(varying)}."
        few = split_param(varying)
        if varying[few] > SPLIT_MAX:
            return (f"The split view needs a param with {SPLIT_MAX} values or fewer.\n"
                    f"The fewest is {few} with {varying[few]}. Narrow it, or use the 3D point map.")
        return None
    if n > 2:
        hint = "Use the 3D point map or split view, or constrain one param to a single value." if n == 3 \
            else f"Constrain {n - 2} more param(s) to a single value."
        return f"{n} params vary: {varying_text(varying)}.\nPlots show at most 2.\n{hint}"
    return None


def build_panel(view, df, metrics, colorby, varying, encoding="both"):
    """Return (figure, title) for one analytics panel."""
    if df is None or len(df) == 0:
        return make_empty("No rows match the current constraints"), ANALYTICS_LABEL[view]
    if not metrics:
        return make_empty("Select a metric"), ANALYTICS_LABEL[view]
    m = metrics[0]
    label = ANALYTICS_LABEL[view]
    view = resolve_auto(view, varying)
    if view == "blocked":
        return make_empty(blocked_reason("auto", varying)), f"{label} · blocked"
    reason = blocked_reason(view, varying)
    if reason:
        return make_empty(reason), f"{label} · blocked"

    dims = list(varying)  # plot axes = the params that actually vary
    if view == "points":
        return fig_points(df, dims, m, colorby, encoding)
    if view == "split":
        return fig_split(df, dict(varying), m)
    if view == "hist":
        return fig_hist(df, m)
    if view == "2d":
        return fig_slices(df, dims, m, colorby)
    if view == "3d":
        return fig_3d(df, dims, m, colorby)
    if view == "heatmap":
        return fig_heatmap(df, dims, m, colorby)
    if view == "sensitivity":
        return fig_sensitivity(df, dims, metrics)
    if view == "pairs":
        return fig_pairs(df, metrics, colorby)
    return make_empty(), label


def color_meaning(view, metrics, colorby, varying, encoding="both"):
    """What the color shows in this panel, or None when color carries no meaning.
    Mirrors the fallbacks in build_panel."""
    if not metrics:
        return None
    m, n = metrics[0], len(varying)
    view = resolve_auto(view, varying)
    if view == "blocked" or blocked_reason(view, varying):
        return None
    if view == "split":
        return f"{m} · one shared scale"
    if view == "points":
        parts = []
        if encoding in ("both", "size"):
            parts.append(f"size = {m}")
        if encoding in ("both", "color") or colorby:
            parts.append(f"color = {colorby or m}")
        return " · ".join(parts)
    if view in ("3d", "heatmap") and n < 2:
        view = "2d"
    if view == "2d" and n == 0:
        view = "hist"

    if view in ("hist", "sensitivity"):
        return None
    if view == "2d":
        if n == 1:
            return colorby if colorby and colorby != m else None
        return f"one line per {list(varying)[1]}"
    if view == "heatmap":
        return f"mean {m}"
    return colorby or m


def panel_card(title, graph, title_id=None, legend=None, view=None):
    extra = {"id": title_id} if title_id else {}
    head = [html.Div(title, className="panel-title", title=title, **extra)]
    if view:
        head.append(html.Button("</> matplotlib", id={"type": "mpl-btn", "index": view},
                                className="mpl-btn", title="Get matplotlib code for this plot"))
    return html.Div(className="panel-card", children=[
        html.Div(head, className="panel-head"),
        legend,
        graph,
    ])


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, (int, float, np.number)) else str(v)


def context_legend(constrained, varying, color_label):
    """A strip that says which params vary, which are constrained, and what the color means."""
    fixed = []
    for col, spec in constrained.items():
        if spec["type"] == "range":
            lo, hi = spec["range"]
            fixed.append(f"{col} ∈ [{_fmt(lo)}, {_fmt(hi)}]")
        else:
            vals = spec.get("values") or []
            shown = ", ".join(_fmt(v) for v in vals[:4]) + (f", … ({len(vals)})" if len(vals) > 4 else "")
            fixed.append(f"{col} = {shown}" if len(vals) == 1 else f"{col} ∈ {{{shown}}}")

    def item(label, text, color):
        return html.Span([html.Span(label, className="legend-key"), html.Span(text, style={"color": color})],
                         className="legend-item")

    parts = [item("VARYING", varying_text(varying), ACCENT),
             item("CONSTRAINED", " · ".join(fixed) or "none", "#e0a458")]
    if color_label:
        parts.append(item("COLOR", color_label, "#d7d7e0"))
    return html.Div(parts, className="panel-legend")


def grid_template(n):
    cols = 1 if n == 1 else 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    return {"gridTemplateColumns": f"repeat({cols}, minmax(0, 1fr))",
            "gridTemplateRows": f"repeat({rows}, minmax(0, 1fr))"}


# ─── LAYOUT ───────────────────────────────────────────────────────────────────

def picker_label(text):
    return html.Div(text, style=SECTION_LABEL_STYLE)


def initial_state():
    """Preload a CSV given on the command line, if any.
    Only for `python app.py file.csv`. Under gunicorn, sys.argv holds gunicorn's args."""
    if __name__ == "__main__" and len(sys.argv) > 1:
        path = sys.argv[1]
        df = pd.read_csv(path)
        return register_df(df), detect_col_types(df), f"{path}  ({len(df):,} rows × {len(df.columns)} cols)"
    return None, {}, ""


INIT_KEY, INIT_TYPES, INIT_NAME = initial_state()

app.layout = dbc.Container(
    fluid=True,
    style={"padding": 0, "height": "100vh", "overflow": "hidden", "backgroundColor": "#0d0f14"},
    children=[
        dcc.Store(id="store-data", data=INIT_KEY),
        dcc.Store(id="store-col-types", data=INIT_TYPES),
        dcc.Store(id="store-constrained", data={}),
        dcc.Store(id="store-rendered-keys", data=None),
        dcc.Store(id="store-selected-row", data=None),
        dcc.Store(id="store-filename", data={"name": INIT_NAME.split("  (")[0], "sample": False}),
        dcc.Store(id="store-mpl", data=None),
        dcc.Download(id="download-nb"),

        dbc.Row(style={"height": "100vh", "margin": 0}, children=[

            # ── LEFT PANEL: dimensions ───────────────────────────────────────
            dbc.Col(width=3, className="left-panel", style={
                "height": "100vh", "overflowY": "auto", "borderRight": "1px solid #222",
                "padding": "14px 12px", "display": "flex", "flexDirection": "column", "gap": "4px",
            }, children=[
                html.H5("Dimensions", style={"color": ACCENT, "marginBottom": "10px", "marginTop": "2px"}),

                dcc.Upload(
                    id="upload-csv",
                    children=html.Div(["Drop CSV or ", html.A("Browse", style={"color": ACCENT})]),
                    style={
                        "border": "1px dashed #444", "borderRadius": "5px", "padding": "10px",
                        "textAlign": "center", "cursor": "pointer", "fontSize": "12px", "marginBottom": "6px",
                    },
                    accept=".csv",
                ),
                html.A("How to format your CSV", id="help-open", href="#",
                       style={"fontSize": "11px", "color": ACCENT, "marginBottom": "6px"}),
                dbc.Modal(id="help-modal", size="lg", scrollable=True, is_open=False, children=[
                    dbc.ModalHeader(dbc.ModalTitle("CSV format guide")),
                    dbc.ModalBody(dcc.Markdown(HELP_MD, className="help-md")),
                ]),
                dbc.Modal(id="mpl-modal", size="xl", scrollable=True, is_open=False, children=[
                    dbc.ModalHeader(dbc.ModalTitle("matplotlib code")),
                    dbc.ModalBody([
                        html.Div(id="mpl-note", className="mpl-note"),
                        html.Div(className="mpl-chunk-head", children=[
                            html.Span("1 · Load the CSV"),
                            dcc.Clipboard(id="mpl-copy-load", className="mpl-copy", title="Copy"),
                        ]),
                        html.Pre(id="mpl-load", className="mpl-code"),
                        html.Div(className="mpl-chunk-head", children=[
                            html.Span("2 · Make the figure"),
                            dcc.Clipboard(id="mpl-copy-fig", className="mpl-copy", title="Copy"),
                        ]),
                        html.Pre(id="mpl-fig", className="mpl-code"),
                    ]),
                    dbc.ModalFooter([
                        html.Span(["Copy both as one cell ",
                                   dcc.Clipboard(id="mpl-copy-all", className="mpl-copy", title="Copy both")],
                                  className="mpl-copy-all"),
                        dbc.Button("Download .ipynb", id="mpl-download", color="info", size="sm"),
                    ]),
                ]),
                dcc.Dropdown(id="sample-select", options=list_samples(), placeholder="…or pick a sample",
                             clearable=False, style={"fontSize": "12px", "marginBottom": "6px"}),
                html.Div(id="filename-display", children=INIT_NAME,
                         style={"fontSize": "11px", "color": "#9aa0ac", "marginBottom": "4px"}),

                dbc.Accordion(id="col-types-accordion", children=[
                    dbc.AccordionItem(html.Div(id="col-type-panel"), title="Column types",
                                      item_id="col-types-item"),
                ], start_collapsed=True, flush=True, style={"fontSize": "12px", "marginBottom": "6px"}),

                html.Hr(style={"borderColor": "#222", "margin": "6px 0"}),

                picker_label("UNCONSTRAINED  ·  drag down to constrain"),
                html.Div(id="unconstrained-list", className="drop-zone",
                         **{"data-zone": "free"}, style={"minHeight": "60px"}),

                html.Hr(style={"borderColor": "#222", "margin": "6px 0"}),

                picker_label("CONSTRAINED  ·  drag up to free"),
                html.Div(id="constrained-list", className="drop-zone",
                         **{"data-zone": "constrained"}, style={"minHeight": "80px", "flex": "1"}),
            ]),

            # ── RIGHT PANEL: analytics ───────────────────────────────────────
            dbc.Col(width=9, style={
                "height": "100vh", "padding": "12px 14px", "display": "flex", "flexDirection": "column",
            }, children=[
                dbc.Row([
                    dbc.Col([
                        picker_label("METRICS  ·  first checked = focus"),
                        dcc.Checklist(id="metric-select", options=[], value=[], className="picker-box"),
                    ], width=4),
                    dbc.Col([
                        picker_label("ANALYTICS"),
                        dcc.Checklist(id="analytics-select", options=ANALYTICS, value=["auto"],
                                      className="picker-box"),
                    ], width=4),
                    dbc.Col([
                        picker_label("COLOR BY"),
                        dcc.Dropdown(id="colorby-select", placeholder="(focus metric)",
                                     style={"fontSize": "12px"}),
                        picker_label("POINT MAP  ·  metric shown as"),
                        dcc.RadioItems(
                            id="point-encoding",
                            options=[{"label": "size + color", "value": "both"},
                                     {"label": "size", "value": "size"},
                                     {"label": "color", "value": "color"}],
                            value="both", inline=True, className="picker-inline",
                        ),
                        picker_label("ROWS"),
                        html.Div(id="point-count", style={"fontSize": "12px", "color": "#b4b9c4"}),
                    ], width=4),
                ], style={"marginBottom": "10px", "flexShrink": 0}),

                html.Div(id="panel-grid", className="panel-grid"),
            ]),
        ]),
    ],
)

# ─── CALLBACKS: loading and column types ──────────────────────────────────────

@app.callback(
    Output("help-modal", "is_open"),
    Input("help-open", "n_clicks"),
    prevent_initial_call=True,
)
def open_help(_):
    return True


@app.callback(
    Output("store-data", "data"),
    Output("store-col-types", "data"),
    Output("store-constrained", "data"),
    Output("store-selected-row", "data"),
    Output("filename-display", "children"),
    Output("col-types-accordion", "active_item"),
    Output("store-filename", "data"),
    Input("upload-csv", "contents"),
    Input("sample-select", "value"),
    State("upload-csv", "filename"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def parse_csv(contents, sample, filename, old_key):
    try:
        if ctx.triggered_id == "sample-select":
            if sample not in list_samples():  # only files from the samples folder
                return (no_update,) * 7
            filename = sample
            df = pd.read_csv(SAMPLES_DIR / sample)
        elif contents:
            _, content_string = contents.split(",", 1)
            df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode("utf-8-sig")))
        else:
            return (no_update,) * 7
    except Exception as e:  # show the error; keep the old data
        return (no_update, no_update, no_update, no_update, f"Could not read {filename}: {e}",
                no_update, no_update)
    DATASETS.pop(old_key, None)
    key = register_df(df)
    # Open the column-type panel so the user checks the guesses first.
    return (key, detect_col_types(df), {}, None,
            f"{filename}  ({len(df):,} rows × {len(df.columns)} cols)", "col-types-item",
            {"name": filename, "sample": ctx.triggered_id == "sample-select"})


@app.callback(
    Output("col-type-panel", "children"),
    Input("store-data", "data"),
    State("store-col-types", "data"),
)
def render_col_type_panel(key, col_types):
    if not col_types:
        return html.Div("Load a CSV first.", style={"color": "#8a8f9c", "fontSize": "12px"})
    return [
        dbc.Row([
            dbc.Col(html.Div(col, title=col, style={
                "fontSize": "11px", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }), width=6),
            dbc.Col(dcc.Dropdown(
                id={"type": "col-type-dropdown", "index": col},
                options=["parameter", "metric", "result", "ignore"],
                value=ctype, clearable=False, style={"fontSize": "11px"},
            ), width=6),
        ], className="mb-1", align="center")
        for col, ctype in col_types.items()
    ]


@app.callback(
    Output("store-col-types", "data", allow_duplicate=True),
    Input({"type": "col-type-dropdown", "index": ALL}, "value"),
    State({"type": "col-type-dropdown", "index": ALL}, "id"),
    State("store-col-types", "data"),
    prevent_initial_call=True,
)
def update_col_types(values, ids, col_types):
    if not col_types or not ids:
        return no_update
    updated = dict(col_types)
    for id_dict, val in zip(ids, values):
        updated[id_dict["index"]] = val
    return no_update if updated == col_types else updated


@app.callback(
    Output("metric-select", "options"),
    Output("metric-select", "value"),
    Output("colorby-select", "options"),
    Input("store-col-types", "data"),
    State("metric-select", "value"),
)
def update_pickers(col_types, current):
    if not col_types:
        return [], [], []
    metrics = [c for c, t in col_types.items() if t == "metric"]
    params = [c for c, t in col_types.items() if t == "parameter"]
    kept = [m for m in (current or []) if m in metrics]
    return metrics, kept or metrics[:1], metrics + params


# ─── CALLBACKS: dimensions ────────────────────────────────────────────────────

@app.callback(
    Output("unconstrained-list", "children"),
    Output("constrained-list", "children"),
    Output("store-rendered-keys", "data"),
    Input("store-col-types", "data"),
    Input("store-constrained", "data"),
    State("store-data", "data"),
    State("store-rendered-keys", "data"),
)
def render_param_lists(col_types, constrained, key, rendered):
    if not col_types:
        return html.Div("No data.", style={"color": "#8a8f9c", "fontSize": "12px"}), html.Div(), None

    param_cols = [c for c, t in col_types.items() if t == "parameter"]
    constrained = constrained or {}
    constrained_cols = [c for c in param_cols if c in constrained]
    unconstrained_cols = [c for c in param_cols if c not in constrained]

    # A slider move only changes values, not which params are constrained.
    # Skip the re-render then, so the slider is not rebuilt under the mouse.
    signature = {"params": param_cols, "constrained": constrained_cols}
    if ctx.triggered_id == "store-constrained" and rendered == signature:
        return no_update, no_update, no_update

    df = get_df(key)

    unc_items = [
        html.Div([
            html.Span("⠿ ", style={"color": "#a3a9b6", "marginRight": "4px"}),
            html.Span(col, style={"fontSize": "12px", "overflow": "hidden",
                                  "textOverflow": "ellipsis", "whiteSpace": "nowrap", "flex": "1"}),
            html.Span(
                _n_values_label(df, col), style={"color": "#9aa0ac", "fontSize": "10px", "marginLeft": "6px"},
            ),
            html.Span("→", id={"type": "constrain-btn", "index": col}, n_clicks=0,
                      title="Constrain", **{"data-action": "constrain", "data-btn-col": col},
                      style={"cursor": "pointer", "color": "#4fd1e6", "marginLeft": "8px", "fontSize": "13px"}),
        ], className="param-chip", **{"data-col": col, "data-from": "free"}, style=BADGE_STYLE)
        for col in unconstrained_cols
    ]
    unc_div = unc_items or html.Div("(none free)", style={"color": "#8a8f9c", "fontSize": "12px"})

    con_items = []
    for col in constrained_cols:
        spec = constrained[col]
        vals = df[col].dropna() if df is not None and col in df.columns else pd.Series([0.0, 1.0])

        if spec["type"] == "range":
            col_min, col_max = float(vals.min()), float(vals.max())
            step = (col_max - col_min) / 200 if col_max != col_min else 0.01
            control = html.Div(dcc.RangeSlider(
                id={"type": "constraint-range", "index": col},
                min=col_min, max=col_max, value=spec["range"], step=step,
                marks={col_min: f"{col_min:.3g}", col_max: f"{col_max:.3g}"},
                tooltip={"placement": "bottom", "always_visible": False},
            ), style={"paddingTop": "6px"})
        else:
            unique_vals = sorted(vals.unique().tolist())
            control = dcc.Dropdown(
                id={"type": "constraint-values", "index": col},
                options=[{"label": f"{v:.4g}" if isinstance(v, float) else str(v), "value": v}
                         for v in unique_vals],
                value=spec.get("values", unique_vals), multi=True,
                placeholder="Select values…", style={"fontSize": "11px"},
            )

        # Only the header is draggable, so dragging a slider does not move the card.
        header = html.Div([
            html.Span("⠿ ", style={"color": "#a3a9b6", "marginRight": "4px"}),
            html.Span(col, style={"fontSize": "12px", "fontWeight": "600", "overflow": "hidden",
                                  "textOverflow": "ellipsis", "flex": "1", "whiteSpace": "nowrap"}),
            html.Span("range" if spec["type"] == "range" else "values",
                      id={"type": "mode-btn", "index": col}, n_clicks=0,
                      title="Switch between a range and specific values",
                      style={"cursor": "pointer", "color": "#b4b9c4", "fontSize": "10px", "marginLeft": "8px",
                             "border": "1px solid #3a4255", "borderRadius": "3px", "padding": "0 4px"}),
            html.Span("← free", id={"type": "free-btn", "index": col}, n_clicks=0,
                      title="Release constraint", **{"data-action": "free", "data-btn-col": col},
                      style={"cursor": "pointer", "color": "#ff7b7b", "fontSize": "11px", "marginLeft": "8px"}),
        ], className="param-chip", **{"data-col": col, "data-from": "constrained"},
            style={"padding": "5px 8px", "backgroundColor": "#1a1f2e", "display": "flex", "alignItems": "center"})

        con_items.append(html.Div([
            header,
            html.Div(control, style={"padding": "6px 8px 8px 8px"}),
        ], style={"marginBottom": "6px", "border": "1px solid #2a3040", "borderRadius": "5px"}))

    con_div = con_items or html.Div("(none constrained)", style={"color": "#8a8f9c", "fontSize": "12px"})
    return unc_div, con_div, signature


def _n_values_label(df, col):
    if df is None or col not in df.columns:
        return ""
    return f"{df[col].nunique()} vals"


def pick_trigger(triggered):
    """Find the real user action among the triggered inputs, as (type, column, value).

    After the lists redraw, Dash can report every new button as changed in one
    call, each with n_clicks=0. A button with 0 clicks was not clicked, so skip
    it and look for the entry that was."""
    for t in triggered or []:
        prop_id, value = t.get("prop_id", ""), t.get("value")
        if not prop_id.startswith("{"):
            continue
        tid = json.loads(prop_id.rsplit(".", 1)[0])
        if tid["type"].endswith("-btn") and not value:
            continue
        return tid["type"], tid["index"], value
    return None


@app.callback(
    Output("store-constrained", "data", allow_duplicate=True),
    Input({"type": "constrain-btn", "index": ALL}, "n_clicks"),
    Input({"type": "free-btn", "index": ALL}, "n_clicks"),
    Input({"type": "mode-btn", "index": ALL}, "n_clicks"),
    Input({"type": "constraint-range", "index": ALL}, "value"),
    Input({"type": "constraint-values", "index": ALL}, "value"),
    State("store-constrained", "data"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def update_constrained(_c, _f, _m, _r, _v, constrained, key):
    constrained = constrained or {}
    trig = pick_trigger(ctx.triggered)
    if trig is None:
        return no_update
    ttype, col, value = trig
    df = get_df(key)

    if ttype == "constrain-btn":
        if col in constrained or df is None:
            return no_update
        return {**constrained, col: default_constraint(df[col].dropna())}

    if ttype == "free-btn":
        return {k: v for k, v in constrained.items() if k != col}

    if ttype == "mode-btn" and col in constrained and df is not None:
        vals = df[col].dropna()
        if constrained[col]["type"] == "range":
            spec = {"type": "values", "values": sorted(vals.unique().tolist())}
        elif pd.api.types.is_numeric_dtype(vals):
            spec = {"type": "range", "range": [float(vals.min()), float(vals.max())]}
        else:
            return no_update  # text columns only support specific values
        return {**constrained, col: spec}

    if ttype == "constraint-range" and col in constrained:
        return {**constrained, col: {"type": "range", "range": value}}

    if ttype == "constraint-values" and col in constrained:
        return {**constrained, col: {"type": "values", "values": value or []}}

    return no_update


# ─── CALLBACKS: analytics ─────────────────────────────────────────────────────

@app.callback(
    Output("panel-grid", "children"),
    Output("panel-grid", "style"),
    Output("point-count", "children"),
    Input("store-data", "data"),
    Input("store-col-types", "data"),
    Input("store-constrained", "data"),
    Input("metric-select", "value"),
    Input("analytics-select", "value"),
    Input("colorby-select", "value"),
    Input("point-encoding", "value"),
    State("store-selected-row", "data"),
)
def render_panels(key, col_types, constrained, metrics, views, colorby, encoding, selected_row):
    df = get_df(key)
    if df is None or not col_types:
        return [panel_card("", dcc.Graph(figure=make_empty(), style={"height": "100%"}))], grid_template(1), ""

    constrained = constrained or {}
    filtered = apply_constraints(df, constrained)
    param_cols = [c for c, t in col_types.items() if t == "parameter"]
    free_params = [c for c in param_cols if c not in constrained]
    metrics = [m for m in (metrics or []) if m in df.columns]
    views = [v for v in (views or []) if v in ANALYTICS_LABEL] or ["auto"]

    varying = varying_params(filtered, param_cols, free_params)
    count = f"{len(filtered):,} / {len(df):,} rows · {len(varying)} varying param(s)"
    if len(varying) > 2:
        count += " · most plots need ≤ 2"

    cards = []
    for view in views:
        if view == "inspector":
            fig, title = fig_inspector(df, selected_row, col_types, param_cols, metrics)
            legend = None
            graph = dcc.Graph(id="inspector-graph", figure=style_fig(fig),
                              style={"height": "100%"}, config={"displaylogo": False})
        else:
            fig, title = build_panel(view, filtered, metrics, colorby, varying, encoding)
            legend = context_legend(constrained, varying,
                                    color_meaning(view, metrics, colorby, varying, encoding))
            # Keep zoom and 3D camera while constraints change, but reset when the axes change.
            uirev = f"{view}|{'|'.join(free_params)}|{'|'.join(varying)}|{metrics[:1]}"
            graph = dcc.Graph(id={"type": "panel", "index": view}, figure=style_fig(fig, uirev),
                              style={"height": "100%"},
                              config={"displaylogo": False, "scrollZoom": True})
        cards.append(panel_card(title, graph, "inspector-title" if view == "inspector" else None, legend,
                                view=view if metrics or view == "inspector" else None))

    return cards, grid_template(len(cards)), count


@app.callback(
    Output("store-selected-row", "data", allow_duplicate=True),
    Output("analytics-select", "value", allow_duplicate=True),
    Input({"type": "panel", "index": ALL}, "clickData"),
    State("analytics-select", "value"),
    State("store-data", "data"),
    State("store-col-types", "data"),
    State("store-constrained", "data"),
    prevent_initial_call=True,
)
def select_row(_clicks, views, key, col_types, constrained):
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update, no_update
    pts = ctx.triggered[0]["value"].get("points") or []
    if not pts:
        return no_update, no_update
    pt = pts[0]
    row = pt.get("customdata")
    if isinstance(row, list):
        row = row[0] if row else None
    if row is None or (isinstance(row, float) and np.isnan(row)):
        # Some traces (for example 3D surfaces) do not send customdata.
        # Find the nearest run from the clicked x and y instead.
        row = row_from_xy(pt, key, col_types, constrained or {})
    if row is None:
        return no_update, no_update

    # Show the inspector if the data has result columns and it is not open yet.
    views = list(views or [])
    has_results = any(t == "result" for t in (col_types or {}).values())
    if has_results and "inspector" not in views:
        return int(row), views + ["inspector"]
    return int(row), no_update


def row_from_xy(pt, key, col_types, constrained):
    df = get_df(key)
    if df is None or "x" not in pt or "y" not in pt:
        return None
    params = [c for c, t in col_types.items() if t == "parameter"]
    sub = apply_constraints(df, constrained)
    dims = list(varying_params(sub, params, [c for c in params if c not in constrained]))
    if len(dims) < 2:
        return None
    p1, p2 = dims[0], dims[1]
    try:
        d = (numeric_view(sub[p1]) - float(pt["x"])) ** 2 + (numeric_view(sub[p2]) - float(pt["y"])) ** 2
    except (TypeError, ValueError):
        return None
    return int(sub.index[int(np.argmin(d))]) if len(sub) else None


@app.callback(
    Output("inspector-graph", "figure"),
    Output("inspector-title", "children"),
    Input("store-selected-row", "data"),
    State("store-data", "data"),
    State("store-col-types", "data"),
    State("metric-select", "value"),
    prevent_initial_call=True,
)
def update_inspector(row, key, col_types, metrics):
    df = get_df(key)
    if df is None or not col_types:
        return no_update, no_update
    param_cols = [c for c, t in col_types.items() if t == "parameter"]
    fig, title = fig_inspector(df, row, col_types, param_cols, metrics or [])
    return style_fig(fig), title


# ─── CALLBACKS: matplotlib export ─────────────────────────────────────────────

def figure_code(view, filtered, df_full, metrics, colorby, varying, encoding, title,
                col_types, selected_row):
    """matplotlib code lines for one panel, and None; or None and the reason. Mirrors build_panel."""
    if view == "inspector":
        result_cols = [c for c, t in col_types.items() if t == "result" and c in df_full.columns][:4]
        if not result_cols:
            return None, "No result columns. Mark a column as 'result' under Column types."
        if selected_row is None or selected_row not in df_full.index:
            return None, "Click a point in a plot first. The code draws the results of that run."
        return mpl_export.code_inspector(int(selected_row), result_cols, title), None
    if len(filtered) == 0:
        return None, "No rows match the current constraints."
    if not metrics:
        return None, "Select a metric first."
    m = metrics[0]
    view = resolve_auto(view, varying)
    reason = blocked_reason("auto" if view == "blocked" else view, varying)
    if reason:
        return None, reason
    dims = list(varying)
    if view == "points":
        return mpl_export.code_points(filtered, dims, m, colorby, encoding, title), None
    if view == "split":
        split = split_param(varying)
        x, y = [p for p in varying if p != split][:2]
        return mpl_export.code_split(filtered, split, x, y, m, title), None
    if view == "hist":
        return mpl_export.code_hist(filtered, m, title), None
    if view == "2d":
        return mpl_export.code_slices(filtered, dims, m, colorby, title), None
    if view == "3d":
        return mpl_export.code_3d(filtered, dims, m, colorby, title), None
    if view == "heatmap":
        return mpl_export.code_heatmap(filtered, dims, m, colorby, title), None
    if view == "sensitivity":
        if not dims:
            return None, "No varying parameters. Release a constraint."
        return mpl_export.code_sensitivity(dims, metrics, title), None
    if view == "pairs":
        if len(metrics) < 2:
            return None, "Select 2 or more metrics."
        return mpl_export.code_pairs(filtered, metrics, colorby, title), None
    return None, "This view has no matplotlib export."


@app.callback(
    Output("mpl-modal", "is_open"),
    Output("mpl-note", "children"),
    Output("mpl-load", "children"),
    Output("mpl-fig", "children"),
    Output("mpl-copy-load", "content"),
    Output("mpl-copy-fig", "content"),
    Output("mpl-copy-all", "content"),
    Output("store-mpl", "data"),
    Input({"type": "mpl-btn", "index": ALL}, "n_clicks"),
    State("store-data", "data"),
    State("store-col-types", "data"),
    State("store-constrained", "data"),
    State("metric-select", "value"),
    State("colorby-select", "value"),
    State("point-encoding", "value"),
    State("store-selected-row", "data"),
    State("store-filename", "data"),
    prevent_initial_call=True,
)
def export_matplotlib(_clicks, key, col_types, constrained, metrics, colorby, encoding, selected_row, fileinfo):
    # New panels fire this with n_clicks=None. Only react to a real click.
    if not ctx.triggered or not ctx.triggered[0]["value"] or not isinstance(ctx.triggered_id, dict):
        return (no_update,) * 8
    df = get_df(key)
    if df is None or not col_types:
        return True, "The data is gone from the server. Load the CSV again.", "", "", "", "", "", None

    view = ctx.triggered_id["index"]
    constrained = constrained or {}
    filtered = apply_constraints(df, constrained)
    param_cols = [c for c, t in col_types.items() if t == "parameter"]
    free_params = [c for c in param_cols if c not in constrained]
    metrics = [m for m in (metrics or []) if m in df.columns]
    varying = varying_params(filtered, param_cols, free_params)

    if view == "inspector":
        _, title = fig_inspector(df, selected_row, col_types, param_cols, metrics)
    else:
        _, title = build_panel(view, filtered, metrics, colorby, varying, encoding)
    lines, reason = figure_code(view, filtered, df, metrics, colorby, varying, encoding, title,
                                col_types, selected_row)
    if lines is None:
        return True, reason, "", "", "", "", "", None

    fileinfo = fileinfo or {}
    load = mpl_export.load_code(fileinfo.get("name") or "data.csv", fileinfo.get("sample"), constrained)
    fig = "\n".join(lines)
    both = load + "\n\n\n" + fig
    note = ("Paste chunk 1 into a notebook cell and run it. Paste chunk 2 into the next cell. "
            "Or download both as a notebook.")
    return True, note, load, fig, load, fig, both, {"load": load, "fig": fig, "title": title}


@app.callback(
    Output("download-nb", "data"),
    Input("mpl-download", "n_clicks"),
    State("store-mpl", "data"),
    prevent_initial_call=True,
)
def download_notebook(_, code):
    if not code:
        return no_update
    nb = mpl_export.notebook(code["load"], code["fig"], code["title"])
    return dict(content=nb, filename="parameter_explorer_figure.ipynb")


# ─── ENTRY ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "1") == "1",
            host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8050")))
