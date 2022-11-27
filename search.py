import requests
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from scrape import parse_content_from_bs


def read_file(filename: str) -> pd.DataFrame:
    if filename.endswith(".pq"):
        return pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        return pd.read_csv(filename)
    else:
        raise ValueError("Invalid file extension! Must be .pq or .csv")


def unpack_text(text: str) -> str:
    return text.replace(";", " ")


def wiki_page_from_url_to_content_text(url: str) -> str:
    bs = BeautifulSoup(requests.get(url).text, "html.parser")
    return parse_content_from_bs(bs)[1]


class SimpleSearcheEngine:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, stop_words="english")
        self.index = pd.DataFrame()
        self.acceptance_mask = pd.Series(dtype=bool)

    def compile_index(
        self, documents: list[str], urls: list[str], acceptance_threshold: float = 0
    ) -> None:
        transformed_docs = self.vectorizer.fit_transform(documents).toarray()
        self.index = pd.DataFrame(
            transformed_docs,
            index=urls,
            columns=self.vectorizer.get_feature_names_out(),
        )
        self.acceptance_mask = self.index.sum(axis=0) >= acceptance_threshold
        self.index = self.index.loc[:, self.acceptance_mask]

    def make_scores_dataframe(
        self, multipliers: np.ndarray, top_n: int
    ) -> pd.DataFrame:
        scores_df = self.index * multipliers
        scores_df["_SCORE_TOTAL"] = scores_df.sum(axis=1)
        top_scores_df = scores_df.sort_values(by="_SCORE_TOTAL", ascending=False).head(
            top_n
        )
        return top_scores_df.loc[:, (top_scores_df != 0).any(axis=0)]

    def vectorize_text(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()[:, self.acceptance_mask]

    def get_similarities(self, vectorized_query: np.ndarray) -> pd.Series:
        similarities = self.index.dot(vectorized_query.T).squeeze()
        divisor = np.linalg.norm(self.index, axis=1) * np.linalg.norm(vectorized_query)
        cosine_similarities = similarities / divisor
        return cosine_similarities

    def search_single(self, query: str, top_n: int = 10) -> list[str]:
        vectorized_query = self.vectorize_text([query])
        return self.get_similarities(vectorized_query).nlargest(top_n).index.to_list()

    def search_multiple(self, queries: list[str], top_n: int = 10) -> list[list[str]]:
        vectorized_queries = self.vectorize_text(queries)
        similarities_list = (
            [
                self.get_similarities(vectorized_query)
                for vectorized_query in vectorized_queries
            ],
        )
        return (
            pd.concat(*similarities_list, axis=1)
            .mean(axis=1)
            .nlargest(top_n)
            .index.to_list()
        )

    def get_scores_dataframe_from_query(
        self, query: str, top_n: int = 10
    ) -> pd.DataFrame:
        query_extended = np.broadcast_to(self.vectorize_text([query]), self.index.shape)
        return self.make_scores_dataframe(query_extended, top_n)

    def get_scores_dataframe_from_queries(
        self, queries: list[str], top_n: int = 10
    ) -> pd.DataFrame:
        vectorized_queries = self.vectorize_text(queries)
        queries_extended = np.broadcast_to(
            vectorized_queries.mean(axis=0), self.index.shape
        )
        return self.make_scores_dataframe(queries_extended, top_n)
