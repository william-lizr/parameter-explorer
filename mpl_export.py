"""Turn an explorer panel into matplotlib code for a Jupyter notebook.

The code comes in two chunks:
  load chunk    read the CSV and apply the explorer's constraints -> `df`
  figure chunk  draw the panel from `df` with matplotlib

The app itself does not need matplotlib. It only writes the code as text.
"""
import json

import numpy as np
import pandas as pd

ACCENT = "#17a2b8"
REPO_RAW = "https://raw.githubusercontent.com/william-lizr/parameter-explorer/main/samples/"

q = repr  # a Python literal for a column name or value


def _is_num(df, col):
    return pd.api.types.is_numeric_dtype(df[col])


# ─── LOAD CHUNK ───────────────────────────────────────────────────────────────

def load_code(filename, is_sample, constrained):
    path = REPO_RAW + filename if is_sample else filename
    note = "  # a sample file from the repo" if is_sample else "  # change this to the path of your file"
    lines = [
        "import json",
        "import re",
        "",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "",
        f"CSV_PATH = {q(path)}{note}",
        "df_all = pd.read_csv(CSV_PATH)",
        "",
        "",
        "def as_num(s):",
        '    """Numbers for plotting. Text columns become category codes 0, 1, 2 …"""',
        "    if pd.api.types.is_numeric_dtype(s):",
        "        return s.astype(float)",
        "    return s.astype(\"category\").cat.codes.astype(float)",
        "",
        "",
        "# Constraints set in the explorer",
        "df = df_all",
    ]
    for col, spec in (constrained or {}).items():
        if spec["type"] == "range":
            lo, hi = spec["range"]
            lines.append(f"df = df[df[{q(col)}].between({lo!r}, {hi!r})]")
        else:
            lines.append(f"df = df[df[{q(col)}].isin({list(spec.get('values') or [])!r})]")
    if not constrained:
        lines.append("# (none)")
    lines.append('print(f"{len(df):,} of {len(df_all):,} rows")')
    return "\n".join(lines)


# ─── FIGURE CHUNKS ────────────────────────────────────────────────────────────

def _text_axes(df, cols):
    """For text params on an axis: keep the labels, then plot them as codes."""
    lines = []
    for c in cols:
        if not _is_num(df, c):
            v = _var(c)
            lines += [f"{v}_labels = df[{q(c)}].astype(\"category\").cat.categories  # text param",
                      f"df = df.assign(**{{{q(c)}: as_num(df[{q(c)}])}})"]
    return lines


def _ticks(df, col, axis, ax="ax"):
    """Put the text labels back on the axis."""
    if _is_num(df, col):
        return []
    v = _var(col)
    return [f"{ax}.set_{axis}ticks(range(len({v}_labels)), {v}_labels)"]


def _var(col):
    """A safe Python name made from a column name."""
    s = "".join(ch if ch.isalnum() else "_" for ch in str(col))
    return s if s and not s[0].isdigit() else "c_" + s


def _finish(title):
    return [f"fig.suptitle({q(title)}, fontsize=10)", "plt.show()"]


def code_hist(df, metric, title):
    return [
        "fig, ax = plt.subplots(figsize=(6, 4), layout=\"constrained\")",
        f"ax.hist(df[{q(metric)}].dropna(), bins=40, color={q(ACCENT)}, edgecolor=\"#0d6e7e\")",
        f"ax.set_xlabel({q(metric)})",
        "ax.set_ylabel(\"Count\")",
        *_finish(title),
    ]


def code_slices(df, dims, metric, colorby, title):
    if not dims:
        return code_hist(df, metric, title)
    p1 = dims[0]
    head = _text_axes(df, [p1])

    if len(dims) == 1 and colorby and colorby in df.columns and colorby != metric:
        return head + [
            "fig, ax = plt.subplots(figsize=(6, 4), layout=\"constrained\")",
            f"sc = ax.scatter(df[{q(p1)}], df[{q(metric)}], c=as_num(df[{q(colorby)}]), cmap=\"viridis\", s=12)",
            f"fig.colorbar(sc, ax=ax, label={q(colorby)})",
            f"ax.set_xlabel({q(p1)})",
            f"ax.set_ylabel({q(metric)})",
            *_ticks(df, p1, "x"),
            *_finish(title),
        ]

    if len(dims) == 1:
        return head + [
            f"g = df.groupby({q(p1)})[{q(metric)}].agg([\"mean\", \"std\"])",
            "g[\"std\"] = g[\"std\"].fillna(0)",
            "x = g.index.to_numpy(dtype=float)",
            "",
            "fig, ax = plt.subplots(figsize=(6, 4), layout=\"constrained\")",
            f"ax.fill_between(x, g[\"mean\"] - g[\"std\"], g[\"mean\"] + g[\"std\"], color={q(ACCENT)}, alpha=0.2, lw=0, label=\"± 1 std\")",
            f"ax.plot(x, g[\"mean\"], color={q(ACCENT)}, lw=2, label=\"mean\")",
            f"ax.scatter(df[{q(p1)}], df[{q(metric)}], color={q(ACCENT)}, s=8, alpha=0.5, label=\"single runs\")",
            f"ax.set_xlabel({q(p1)})",
            f"ax.set_ylabel({q(metric)})",
            *_ticks(df, p1, "x"),
            "ax.legend(fontsize=8)",
            *_finish(title),
        ]

    p2 = dims[1]
    return head + [
        f"p2 = {q(p2)}",
        f"levels = sorted(df[{q(p2)}].unique())",
        "if len(levels) > 12:  # too many lines to read: keep 12 evenly spaced levels",
        "    levels = [levels[i] for i in np.linspace(0, len(levels) - 1, 12).round().astype(int)]",
        "colors = plt.cm.plasma(np.linspace(0, 0.9, len(levels)))",
        "",
        "fig, ax = plt.subplots(figsize=(6, 4), layout=\"constrained\")",
        "for lvl, col in zip(levels, colors):",
        f"    g = df[df[{q(p2)}] == lvl].groupby({q(p1)})[{q(metric)}].mean()",
        "    ax.plot(g.index, g.values, \"-o\", ms=3, lw=1.5, color=col, label=f\"{p2}={lvl}\")",
        f"ax.set_xlabel({q(p1)})",
        f"ax.set_ylabel({q(metric)})",
        *_ticks(df, p1, "x"),
        "ax.legend(fontsize=7, ncol=2)",
        *_finish(title),
    ]


def _full_grid(df, p1, p2, metric):
    z = df.pivot_table(index=p2, columns=p1, values=metric, aggfunc="mean")
    return bool(z.notna().all().all() and z.shape[0] > 1 and z.shape[1] > 1)


def code_3d(df, dims, metric, colorby, title):
    if len(dims) < 2:
        return code_slices(df, dims, metric, colorby, title)
    p1, p2 = dims[0], dims[1]
    lines = _text_axes(df, [p1, p2])
    if _full_grid(df, p1, p2, metric):
        lines += [
            f"z = df.pivot_table(index={q(p2)}, columns={q(p1)}, values={q(metric)}, aggfunc=\"mean\")",
            "X, Y = np.meshgrid(z.columns.to_numpy(dtype=float), z.index.to_numpy(dtype=float))",
            "",
            "fig = plt.figure(figsize=(7, 5), layout=\"constrained\")",
            "ax = fig.add_subplot(projection=\"3d\")",
            "surf = ax.plot_surface(X, Y, z.to_numpy(), cmap=\"plasma\", edgecolor=\"none\")",
            f"fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label={q(metric)})",
        ]
    else:
        c = colorby if colorby and colorby in df.columns else metric
        lines += [
            "fig = plt.figure(figsize=(7, 5), layout=\"constrained\")",
            "ax = fig.add_subplot(projection=\"3d\")",
            f"sc = ax.scatter(df[{q(p1)}], df[{q(p2)}], df[{q(metric)}], c=as_num(df[{q(c)}]), cmap=\"plasma\", s=6)",
            f"fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label={q(c)})",
        ]
    return lines + [
        f"ax.set_xlabel({q(p1)})",
        f"ax.set_ylabel({q(p2)})",
        f"ax.set_zlabel({q(metric)})",
        *_ticks(df, p1, "x"),
        *_ticks(df, p2, "y"),
        *_finish(title),
    ]


def code_heatmap(df, dims, metric, colorby, title):
    if len(dims) < 2:
        return code_slices(df, dims, metric, colorby, title)
    p1, p2 = dims[0], dims[1]
    return _text_axes(df, [p1, p2]) + [
        f"z = df.pivot_table(index={q(p2)}, columns={q(p1)}, values={q(metric)}, aggfunc=\"mean\")",
        "xs, ys = z.columns.to_numpy(dtype=float), z.index.to_numpy(dtype=float)",
        "",
        "fig, ax = plt.subplots(figsize=(6, 4.5), layout=\"constrained\")",
        "mesh = ax.pcolormesh(xs, ys, z.to_numpy(), cmap=\"plasma\", shading=\"nearest\")",
        f"fig.colorbar(mesh, ax=ax, label={q(metric)})",
        "",
        "# Mark the best cell",
        "iy, ix = np.unravel_index(np.nanargmax(z.to_numpy()), z.shape)",
        f"ax.plot(xs[ix], ys[iy], \"x\", color=\"white\", ms=10, mew=2, label={q('max ' + str(metric))})",
        "ax.legend(fontsize=8, loc=\"upper right\")",
        f"ax.set_xlabel({q(p1)})",
        f"ax.set_ylabel({q(p2)})",
        *_ticks(df, p1, "x"),
        *_ticks(df, p2, "y"),
        *_finish(title),
    ]


def code_points(df, dims, metric, colorby, encoding, title):
    x, y, z = dims[:3]
    color_col = colorby if colorby and colorby in df.columns and colorby not in (x, y, z) else None
    use_size = encoding in ("both", "size")
    use_color = encoding in ("both", "color") or color_col is not None
    c_name = color_col or metric

    lines = _text_axes(df, [x, y, z])
    lines += ["# One point per (x, y, z) cell: the mean over repeated runs"]
    if color_col and color_col != metric:
        lines += [f"g = (df.assign(_color=as_num(df[{q(color_col)}]))",
                  f"       .groupby([{q(x)}, {q(y)}, {q(z)}])",
                  f"       .agg(value=({q(metric)}, \"mean\"), color=(\"_color\", \"mean\")).reset_index())"]
        color_expr = "g[\"color\"]"
    else:
        lines += [f"g = df.groupby([{q(x)}, {q(y)}, {q(z)}])[{q(metric)}].mean().reset_index(name=\"value\")"]
        color_expr = "g[\"value\"]"
    lines += [
        "val = g[\"value\"].to_numpy(dtype=float)",
        "",
        "# Point area grows with the metric",
        "lo, hi = np.nanmin(val), np.nanmax(val)",
        "frac = (val - lo) / (hi - lo) if hi > lo else np.full(val.shape, 0.5)",
        "sizes = 6 + 120 * np.clip(np.nan_to_num(frac), 0, 1)" if use_size else "sizes = 20",
        "",
        "fig = plt.figure(figsize=(7, 5.5), layout=\"constrained\")",
        "ax = fig.add_subplot(projection=\"3d\")",
    ]
    if use_color:
        lines += [f"sc = ax.scatter(g[{q(x)}], g[{q(y)}], g[{q(z)}], s=sizes, c={color_expr}, cmap=\"plasma\", alpha=0.85)",
                  f"fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label={q(c_name)})"]
    else:
        lines += [f"ax.scatter(g[{q(x)}], g[{q(y)}], g[{q(z)}], s=sizes, color={q(ACCENT)}, alpha=0.85)"]
    if use_size:
        lines += [
            "# Size legend",
        f"metric = {q(metric)}",
            "for v in (lo, (lo + hi) / 2, hi):",
            "    f = (v - lo) / (hi - lo) if hi > lo else 0.5",
            "    ax.scatter([], [], [], s=6 + 120 * f, color=\"grey\", label=f\"{metric} = {v:.3g}\")",
            "ax.legend(fontsize=8, loc=\"upper left\")",
        ]
    return lines + [
        f"ax.set_xlabel({q(x)})",
        f"ax.set_ylabel({q(y)})",
        f"ax.set_zlabel({q(z)})",
        *_ticks(df, x, "x"),
        *_ticks(df, y, "y"),
        *_ticks(df, z, "z"),
        *_finish(title),
    ]


def code_split(df, split, x, y, metric, title):
    return _text_axes(df, [x, y]) + [
        f"split, x, y, metric = {q(split)}, {q(x)}, {q(y)}, {q(metric)}",
        "levels = sorted(df[split].dropna().unique())",
        "n = len(levels)",
        "ncols = n if n <= 3 else int(np.ceil(n / 2))",
        "nrows = int(np.ceil(n / ncols))",
        "vmin, vmax = df[metric].min(), df[metric].max()  # one colour scale for all panels",
        "",
        "fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 1, 3 * nrows),",
        "                         sharex=True, sharey=True, squeeze=False, layout=\"constrained\")",
        "for ax, lvl in zip(axes.flat, levels):",
        "    sub = df[df[split] == lvl]",
        "    z = sub.pivot_table(index=y, columns=x, values=metric, aggfunc=\"mean\")",
        "    if z.isna().to_numpy().mean() <= 0.5:  # a grid: draw a heatmap",
        "        im = ax.pcolormesh(z.columns.to_numpy(dtype=float), z.index.to_numpy(dtype=float), z.to_numpy(),",
        "                           cmap=\"plasma\", vmin=vmin, vmax=vmax, shading=\"nearest\")",
        "    else:  # not a grid (for example random search): coloured points",
        "        im = ax.scatter(sub[x], sub[y], c=sub[metric], cmap=\"plasma\", vmin=vmin, vmax=vmax, s=12)",
        "    ax.set_title(f\"{split} = {lvl}\", fontsize=9)",
        *["    " + t for t in _ticks(df, x, "x") + _ticks(df, y, "y")],
        "for ax in axes.flat[n:]:",
        "    ax.set_visible(False)",
        "for ax in axes[-1]:",
        "    ax.set_xlabel(x)",
        "for ax in axes[:, 0]:",
        "    ax.set_ylabel(y)",
        "fig.colorbar(im, ax=axes, label=metric)",
        f"fig.suptitle({q(title)}, fontsize=10)",
        "plt.show()",
    ]


def code_sensitivity(dims, metrics, title):
    metrics, params = metrics[:4], dims[:6]
    return [
        f"metrics = {metrics!r}",
        f"params = {params!r}",
        "",
        "",
        "def eta_squared(x, y):",
        '    """Share of the metric\'s variance explained by this param alone (0 to 1)."""',
        "    total = ((y - y.mean()) ** 2).sum()",
        "    if total == 0:",
        "        return 0.0",
        "    g = y.groupby(x.values)",
        "    return float((g.count() * (g.mean() - y.mean()) ** 2).sum() / total)",
        "",
        "",
        "# Bin params with many distinct values so each group has several rows",
        "groups = {}",
        "for p in params:",
        "    if pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 30:",
        "        groups[p] = pd.cut(df[p], 20).apply(lambda iv: iv.mid).astype(float)",
        "    else:",
        "        groups[p] = df[p]",
        "",
        "fig, axes = plt.subplots(len(metrics), len(params), figsize=(max(2.6 * len(params) + 1, 6), 2.3 * len(metrics) + 0.6),",
        "                         sharey=\"row\", squeeze=False, layout=\"constrained\")",
        "for i, m in enumerate(metrics):",
        "    for j, p in enumerate(params):",
        "        ax = axes[i, j]",
        "        g = df[m].groupby(groups[p].values).agg([\"mean\", \"std\"])",
        "        g[\"std\"] = g[\"std\"].fillna(0)",
        "        if pd.api.types.is_numeric_dtype(g.index):",
        "            x = g.index.to_numpy(dtype=float)",
        "        else:  # text param: plot at 0, 1, 2 … and label the ticks",
        "            x = np.arange(len(g))",
        "            ax.set_xticks(x, g.index, rotation=45)",
        f"        ax.fill_between(x, g[\"mean\"] - g[\"std\"], g[\"mean\"] + g[\"std\"], color={q(ACCENT)}, alpha=0.2, lw=0)",
        f"        ax.plot(x, g[\"mean\"], \"-o\", ms=3, color={q(ACCENT)}, lw=2)",
        "        ax.set_title(f\"{p} · η²={eta_squared(groups[p], df[m]):.2f}\", fontsize=9)",
        "        if j == 0:",
        "            ax.set_ylabel(m)",
        *_finish(title),
    ]


def code_pairs(df, metrics, colorby, title):
    metrics = metrics[:6]
    c_col = colorby if colorby and colorby in df.columns else metrics[0]
    return [
        f"metrics = {metrics!r}",
        f"c = as_num(df[{q(c_col)}])",
        "n = len(metrics)",
        "",
        "fig, axes = plt.subplots(n, n, figsize=(2.2 * n + 1, 2.2 * n), squeeze=False, layout=\"constrained\")",
        "for i in range(n):",
        "    for j in range(n):",
        "        ax = axes[i, j]",
        "        if j > i:  # lower half only",
        "            ax.set_visible(False)",
        "            continue",
        "        if i == j:",
        f"            ax.hist(df[metrics[i]].dropna(), bins=30, color={q(ACCENT)})",
        "        else:",
        "            sc = ax.scatter(df[metrics[j]], df[metrics[i]], c=c, cmap=\"plasma\", s=4, alpha=0.7)",
        "        if i == n - 1:",
        "            ax.set_xlabel(metrics[j])",
        "        if j == 0:",
        "            ax.set_ylabel(metrics[i])",
        f"fig.colorbar(sc, ax=axes, label={q(c_col)})",
        f"fig.suptitle({q(title)}, fontsize=10)",
        "plt.show()",
    ]


def code_inspector(row, result_cols, title):
    return [
        f"ROW = {row}  # row in the CSV (0 = first data row)",
        f"result_cols = {result_cols!r}",
        "rec = df_all.loc[ROW]",
        "",
        "",
        "def parse_result(cell):",
        '    """A result cell as a numeric array: a JSON list, or numbers split by commas, spaces and ";"."""',
        "    if cell is None or (isinstance(cell, float) and np.isnan(cell)):",
        "        return None",
        "    s = str(cell).strip()",
        "    try:",
        "        return np.asarray(json.loads(s), dtype=float)",
        "    except (ValueError, TypeError):",
        "        pass",
        "    rows = [r for r in re.split(r\"[;\\n]\", s) if r.strip()]",
        "    nums = [[float(v) for v in re.findall(r\"[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?\", r)] for r in rows]",
        "    nums = [r for r in nums if r]",
        "    if len(nums) > 1 and len({len(r) for r in nums}) == 1:",
        "        return np.asarray(nums)",
        "    return np.asarray([v for r in nums for v in r]) if nums else None",
        "",
        "",
        "fig, axes = plt.subplots(1, len(result_cols), figsize=(4 * len(result_cols), 3.5),",
        "                         squeeze=False, layout=\"constrained\")",
        "for ax, col in zip(axes[0], result_cols):",
        "    arr = parse_result(rec[col])",
        "    if arr is None:",
        "        ax.set_visible(False)",
        "        continue",
        "    if arr.ndim == 2:  # a matrix",
        "        lim = np.nanmax(np.abs(arr)) or 1",
        "        im = ax.imshow(arr, cmap=\"RdBu_r\", vmin=-lim, vmax=lim)",
        "        fig.colorbar(im, ax=ax, shrink=0.8)",
        "    else:  # a timeseries",
        f"        ax.plot(arr, color={q(ACCENT)}, lw=1.2)",
        "    ax.set_title(col, fontsize=10)",
        *_finish(title),
    ]


# ─── NOTEBOOK ─────────────────────────────────────────────────────────────────

def notebook(load, figure, title):
    """A minimal .ipynb with a title, the load cell and the figure cell."""
    def cell(kind, src):
        c = {"cell_type": kind, "id": f"cell-{kind}-{len(src)}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c

    nb = {
        "cells": [cell("markdown", f"# {title}\n\nMade by Parameter Explorer."),
                  cell("code", load), cell("code", figure)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False)
