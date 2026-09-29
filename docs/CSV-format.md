# CSV format guide

This page tells you how to shape a CSV so the Parameter Explorer can read it.

## The short version

- Use **one row for each simulation run**.
- Use **one column for each parameter, metric, or result**.
- Put the column names in the **first row**.
- Separate values with **commas**. Save the file as **UTF-8**.
- Put a timeseries or matrix **in one cell**, as a JSON list in double quotes.

## A minimal example

```
K,omega,sigma,fc_corr,criticality
0.01,0.01,0.01,-0.0042,0.1928
0.01,0.01,0.05,0.0176,0.1305
0.05,0.01,0.01,0.0310,0.2110
```

Here `K`, `omega` and `sigma` are the inputs of each run. `fc_corr` and `criticality` are the outputs.

## The three column types

Each column has one type. The app guesses the type when you load the file. You can change each guess under **Column types**.

| Type | What it holds | Example |
|---|---|---|
| **parameter** | An input you set for the run. It can be a number or text. | `K`, `tau`, `network` |
| **metric** | One number the run produced. | `fc_corr`, `synchrony` |
| **result** | A list of numbers (a timeseries) or a table of numbers (a matrix) the run produced. | `mean_signal`, `fc_matrix` |
| **ignore** | A column the app must not use. | `notes`, `run_id` |

### How the app guesses

- **Text column** → *parameter*, unless the cell looks like a list of numbers. Then it is a *result*.
- **Number column with few distinct values** (50 or fewer, or fewer than 25% of rows) → *parameter*.
- **Number column with many distinct values** → *metric*.

The guess is wrong sometimes. For example, a metric with only a few possible values looks like a parameter. Always check **Column types** after you load a file.

## Parameters

- Use numbers where you can. Numeric parameters get sliders and real axes.
- Text parameters work too (for example `scale_free`, `llama3-8b`). The app puts them on axes as categories.
- A **grid sweep** (every combination of values) gives the cleanest surfaces and heatmaps.
- A **random search** (values drawn at random) also works. Use the 3D point map or the slices view for it.
- Repeated runs with the same parameters are fine. Add a `seed` column. The app averages repeats where a plot needs one value per cell.

## Metrics

- Each metric cell holds **one number**.
- If a run failed, leave the cell empty or write `NaN`. Do not write other text such as `failed`. Text turns the whole column into text.
- The first metric you tick is the **focus metric**. Most plots show it.

## Results (timeseries and matrices)

A result cell holds a whole array. You see it in the **Result inspector** when you click a point in a plot.

### Timeseries (1D)

Write a JSON list. Put the cell in double quotes, because the list contains commas.

```
K,omega,fc_corr,mean_signal
0.01,0.01,-0.0042,"[0.015, 0.011, 0.164, 0.236, 0.154]"
```

### Matrix (2D)

Write a JSON list of rows. All rows must have the same length.

```
K,fc_corr,fc_matrix
0.01,-0.0042,"[[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]"
```

The app shows a matrix as a heatmap, with red for positive values and blue for negative values.

### Other accepted forms

JSON is best. The app also reads these:

- Values separated by commas or spaces: `"0.1, 0.2, 0.3"` or `"0.1 0.2 0.3"`.
- Matrix rows separated by semicolons: `"1 0.2; 0.2 1"`.

### Keep result cells small

A whole results column goes into the server memory. Round values to 3 or 4 decimals. Down-sample long timeseries to a few hundred points. A file of a few MB loads fast. A file of hundreds of MB does not.

## How to write the CSV from Python

```python
import json
import numpy as np
import pandas as pd

rows = []
for K in np.linspace(0.01, 0.5, 12):
    for omega in np.linspace(0.01, 0.15, 10):
        ts = run_model(K, omega)            # your simulation -> 1D array
        fc = compute_fc(ts)                 # your FC -> 2D array
        rows.append({
            "K": K,
            "omega": omega,
            "fc_corr": float(score(fc)),    # one number -> metric
            "mean_signal": json.dumps(np.round(ts, 3).tolist()),   # 1D -> result
            "fc_matrix": json.dumps(np.round(fc, 3).tolist()),     # 2D -> result
        })

pd.DataFrame(rows).to_csv("my_sweep.csv", index=False)
```

`pandas` puts the quotes around the JSON cells for you.

## How to save from Excel or Google Sheets

- Excel: **File → Save As → CSV UTF-8 (Comma delimited)**.
- Google Sheets: **File → Download → Comma-separated values (.csv)**.
- Check that your locale does not use `;` as the separator or `,` as the decimal mark. The app needs `,` between values and `.` in numbers.

## Checklist before you upload

- [ ] The first row holds column names. Each name is unique.
- [ ] Each row is one run.
- [ ] Parameter columns hold the inputs. Metric columns hold one number each.
- [ ] Result cells are JSON lists in double quotes.
- [ ] The file uses commas, `.` decimals and UTF-8.
- [ ] After the upload, you checked **Column types**.

## Common problems

| What you see | Why | Fix |
|---|---|---|
| "Could not read file.csv" | The file is not valid CSV, or not UTF-8. | Save again as CSV UTF-8. Check the quotes around list cells. |
| A metric shows up as a parameter | It has few distinct values. | Set it to *metric* under **Column types**. |
| A result column shows up as a parameter | Its first non-empty cell does not look like a list. | Set it to *result*, or write the cells as JSON lists. |
| "N params vary … Plots show at most 2" | More than 2 parameters take several values. | Drag parameters to **Constrained** and pick one value, or use the 3D point map. |
| The Result inspector is empty | No column is marked as *result*, or you did not click a point. | Mark the column as *result*. Click a point in a plot. |
| Your data disappeared after a pause | The server went to sleep and cleared its memory. | Load the file again. |

## Examples

The app has sample files in the **…or pick a sample** list. They are in the [`samples/`](https://github.com/william-lizr/parameter-explorer/tree/main/samples) folder of the repo. `generate_sample.py` makes them.

| File | What it shows |
|---|---|
| `hopf_bifurcation_2d.csv` | 2 parameters on a dense grid. Metrics only. The simplest case. |
| `stuart_landau_sweep.csv` | 3 parameters on a grid. A timeseries and a matrix result. |
| `ou_random_search.csv` | 4 parameters from a random search, not a grid. |
| `montbrio_mpr.csv` | 4 parameters with a firing-rate timeseries. |
| `avalanche_criticality.csv` | 3 parameters with a histogram as a result. |
| `disinfo_agents.csv` | Text parameters and repeated seeds. |
