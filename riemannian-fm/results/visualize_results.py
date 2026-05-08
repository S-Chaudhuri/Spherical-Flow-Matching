import matplotlib.pyplot as plt
import pandas as pd


def plot_rows_scatter(
    df,
    x_row,
    y_row,
    *,
    ax= None,
    label_points: bool = True,
    point_labels=None,
    title  = None,
    xlabel = None,
    ylabel = None,
    scatter_kwargs = None,
):
    """Scatter-plot values from two DataFrame rows against each other.

    Expected layout: rows are variables (e.g. metric names) and columns are
    datapoints (e.g. runs). This matches `pd.DataFrame(dict_of_dicts)`.

    Args:
        df: A pandas DataFrame.
        x_row: Row label (df.index) or integer row position (df.iloc).
        y_row: Row label (df.index) or integer row position (df.iloc).
        ax: Optional matplotlib Axes to draw into.
        label_points: If True, annotate points with their column names.
        point_labels: Optional iterable of labels matching the selected columns.
        title/xlabel/ylabel: Optional plot labels.
        scatter_kwargs: Extra kwargs forwarded to `ax.scatter`.

    Returns:
        (fig, ax, plotted_df) where plotted_df has index=datapoints and columns
        ['x', 'y'] after numeric coercion and NaN filtering.
    """

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if isinstance(x_row, int):
        x_series = df.iloc[x_row]
        x_name = str(df.index[x_row])
    else:
        x_series = df.loc[x_row]
        x_name = str(x_row)

    if isinstance(y_row, int):
        y_series = df.iloc[y_row]
        y_name = str(df.index[y_row])
    else:
        y_series = df.loc[y_row]
        y_name = str(y_row)

    common_cols = x_series.index.intersection(y_series.index)
    x = pd.to_numeric(x_series.loc[common_cols], errors="coerce")
    y = pd.to_numeric(y_series.loc[common_cols], errors="coerce")

    valid = x.notna() & y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(x.values, y.values, **scatter_kwargs)

    if label_points:
        labels = list(point_labels) if point_labels is not None else [str(c) for c in x.index]
        if len(labels) != len(x.index):
            raise ValueError(
                f"point_labels length ({len(labels)}) must match number of points ({len(x.index)})."
            )
        for xi, yi, lab in zip(x.values, y.values, labels):
            ax.annotate(str(lab), (xi, yi), fontsize=8, alpha=0.8)

    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or y_name)
    ax.set_title(title or f"{y_name} vs {x_name}")
    ax.grid(True, alpha=0.3)

    plotted = pd.DataFrame({x_name: x, y_name: y})
    return fig, ax, plotted