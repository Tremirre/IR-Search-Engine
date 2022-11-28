import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def visualize_top_scores(top_scores: pd.DataFrame, top: int) -> None:
    for label, series in top_scores.iterrows():
        series = series.sort_values(ascending=False)[1 : top + 1]
        plt.figure(figsize=(10, 5))
        plt.title(label)
        sns.barplot(y=series.index, x=series.values)
        sns.despine(left=True, bottom=True)


def visualize_most_frequent(docs_df: pd.DataFrame, top: int, method="sum") -> None:

    count_vector = CountVectorizer(ngram_range=(1, 1), stop_words="english")
    count_data = count_vector.fit_transform(docs_df.text)
    df = pd.DataFrame(
        count_data.toarray(),
        columns=count_vector.get_feature_names_out(),
        index=docs_df.title,
    )
    if method == "sum":
        words_sum = df.sum(axis=0).sort_values(ascending=False)
        title = f"Top {top} words by total occurrences in articles"
    else:
        words_sum = df.astype(bool).sum(axis=0).sort_values(ascending=False)
        title = f"Top {top} words by number of documents in which they occur"
    if top > 0:
        words_sum = words_sum[:top]
    else:
        words_sum = words_sum[top:]
    words_sum = (
        pd.DataFrame(words_sum)
        .reset_index()
        .rename(columns={0: "amount", "index": "word"})
    )
    plt.figure(figsize=(10, 5))
    plt.title(title)
    sns.barplot(x="amount", y="word", data=words_sum)
    sns.despine(left=True, bottom=True)


def visualize_tfidf(df: pd.DataFrame, top: int) -> None:

    words_sum = df.mean(axis=0).sort_values(ascending=False)
    if top > 0:
        words_sum = words_sum[:top]
    else:
        words_sum = words_sum[top:]
    words_sum = (
        pd.DataFrame(words_sum)
        .reset_index()
        .rename(columns={0: "value", "index": "word"})
    )
    plt.figure(figsize=(10, 5))
    plt.title(f"Top {top} words with highest mean value of tfidf")
    sns.barplot(x="value", y="word", data=words_sum)
    sns.despine(left=True, bottom=True)
