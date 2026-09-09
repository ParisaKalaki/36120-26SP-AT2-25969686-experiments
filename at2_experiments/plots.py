"""Reusable plotting utilities for feature and target exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def explore_distribution(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
) -> None:
    """Print summary stats and plot distribution + time series for a column.

    Used for exploring either the target variable or an individual feature
    on its own, without reference to any other column. Assumes col is
    numeric; for categorical columns, use value_counts().plot(kind="bar")
    instead.
    """
    print(df[col].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    df[col].hist(bins=30, ax=axes[0])
    axes[0].set_xlabel(col)
    axes[0].set_title(f"Distribution of {col}")

    df.set_index(date_col)[col].plot(ax=axes[1])
    axes[1].set_ylabel(col)
    axes[1].set_title(f"{col} Over Time")

    plt.tight_layout()
    plt.show()


def explore_feature_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
) -> None:
    """Plot a feature's relationship with a continuous target and print
    their Pearson correlation.

    Assumes target_col is continuous/numeric. For a categorical target,
    use explore_feature_vs_categorical_target instead, since scatter
    plots and Pearson correlation are not meaningful against categorical
    labels.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature_col], df[target_col], alpha=0.2, s=8)
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.title(f"{feature_col} vs {target_col}")
    plt.show()

    print(df[[feature_col, target_col]].corr())


def explore_feature_vs_categorical_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    class_order: list[str] | None = None,
) -> None:
    """Plot a feature's distribution grouped by a categorical target, and
    print its mutual information score with that target.

    Unlike explore_feature_vs_target (which assumes a continuous target
    and uses scatter plots + Pearson correlation), this function is for
    categorical targets: it uses a grouped boxplot to show how the
    feature's distribution differs across classes, and mutual information
    (rather than a linear-correlation-based measure) to quantify the
    relationship, since it does not assume linearity or monotonicity.

    Parameters
    ----------
    df : pd.DataFrame
    feature_col : str
        Name of the continuous/numeric feature column.
    target_col : str
        Name of the categorical target column.
    class_order : list[str], optional
        Explicit order for the target's categories on the x-axis (e.g.
        ["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"]).
        If not provided, uses the categories' natural sort order.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    plt.figure(figsize=(8, 4))
    df.boxplot(column=feature_col, by=target_col, ax=plt.gca())
    if class_order is not None:
        plt.gca().set_xticks(range(1, len(class_order) + 1))
        plt.gca().set_xticklabels(class_order)
    plt.suptitle("")
    plt.title(f"{feature_col} by {target_col}")
    plt.xlabel(target_col)
    plt.ylabel(feature_col)
    plt.tight_layout()
    plt.show()

    valid = df[[feature_col, target_col]].dropna()
    encoded_target = LabelEncoder().fit_transform(valid[target_col])
    mi = mutual_info_classif(
        valid[[feature_col]], encoded_target, random_state=42
    )[0]
    print(f"Mutual information ({feature_col} vs {target_col}): {mi:.4f}")


def explore_categorical_distribution(
    df: pd.DataFrame,
    col: str,
    class_order: list[str] | None = None,
) -> None:
    """Print value counts (count and percentage) and plot a bar chart for
    a categorical column.

    Used for exploring a categorical target variable's class distribution,
    e.g. to assess class imbalance before modeling.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Name of the categorical column to explore.
    class_order : list[str], optional
        Explicit order for the categories on the x-axis. If not provided,
        uses value_counts()'s natural (frequency-descending) order.
    """
    counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True) * 100

    if class_order is not None:
        counts = counts.reindex(class_order)
        percentages = percentages.reindex(class_order)

    print(pd.DataFrame({"count": counts, "percentage": percentages.round(2)}))

    counts.plot(kind="bar")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=0)
    plt.show()