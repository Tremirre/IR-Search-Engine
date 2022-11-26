import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_top_scores(top_scores: pd.DataFrame, top: int) -> None:
    for label, series in top_scores.iterrows():
        series = series.sort_values(ascending=False)[1 : top + 1]
        plt.figure(figsize=(10, 5))
        plt.title(label)
        sns.barplot(y=series.index, x=series.values)
        sns.despine(left=True, bottom=True)
