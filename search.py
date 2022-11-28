import requests
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from scrape import parse_content_from_bs


def read_file(filename: str) -> pd.DataFrame:
    """
    Reads a file csv or parquet and returns a dataframe

    :param filename: name of the file to be read
    :raises ValueError: if the file is not a csv or parquet file
    :return: pandas dataframe from the file
    """
    if filename.endswith(".pq"):
        return pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        return pd.read_csv(filename)
    else:
        raise ValueError("Invalid file extension! Must be .pq or .csv")


def unpack_text(text: str) -> str:
    """
    Unpacks the text from the dataframe

    :param text: text to be unpacked
    :return: unpacked text
    """
    return text.replace(";", " ")


def wiki_page_from_url_to_content_text(url: str) -> str:
    """
    Fetches a wiki page from a url and returns text in the paragraphs.

    :param url: URL of the wiki page
    :return: text in the paragraphs of the wiki page
    """
    bs = BeautifulSoup(requests.get(url).text, "html.parser")
    return parse_content_from_bs(bs)[1]


class SimpleSearchEngine:
    """
    Class representing a simple search engine that uses TFIDF vectorization.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            use_idf=True, smooth_idf=False, stop_words="english"
        )
        self.index = pd.DataFrame()
        self.acceptance_mask = pd.Series(dtype=bool)

    def compile_index(
        self, documents: list[str], urls: list[str], acceptance_threshold: float = 0
    ) -> None:
        """
        Compiles the index from a list of documents and a list of urls (which serves as index of the dataframe with tfidf scores).

        :param documents: list of documents from which to compile the index
        :param urls: list of urls to corresponding documents
        :param acceptance_threshold: threshold filtering out tokens with lower total tfidf value that the given threshold, defaults to 0
        """
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
        """
        Makes a dataframe with the scores of the documents in the index.

        :param multipliers: array of multipliers for the index
        :param top_n: number of top documents to be returned
        :return: dataframe with index multiplied by the multipliers and sorted by the sum of the scores
        """
        scores_df = self.index * multipliers
        scores_df["_SCORE_TOTAL"] = scores_df.sum(axis=1)
        top_scores_df = scores_df.sort_values(by="_SCORE_TOTAL", ascending=False).head(
            top_n
        )
        return top_scores_df.loc[:, (top_scores_df != 0).any(axis=0)]

    def vectorize_text(self, texts: list[str]) -> np.ndarray:
        """
        Vectorizes a list of texts.
        Filters out tokens that are not in the acceptance mask.

        :param texts: list of texts to be vectorized
        :return: vectorized texts as numpy array
        """
        return self.vectorizer.transform(texts).toarray()[:, self.acceptance_mask]

    def get_similarities(self, vectorized_query: np.ndarray) -> pd.Series:
        """
        Returns a series with the similarities of the documents in the index to the given query.

        :param vectorized_query: vectorized query
        :return: cosine similarities between the documents in the index and the query
        """
        similarities = self.index.dot(vectorized_query.T).squeeze()
        divisor = np.linalg.norm(self.index, axis=1) * np.linalg.norm(vectorized_query)
        cosine_similarities = similarities / divisor
        return cosine_similarities

    def search_single(self, query: str, top_n: int = 10) -> list[str]:
        """
        Search recommendations for a single query

        :param query: query to search for
        :param top_n: number of top recommendations to return, defaults to 10
        :return: list of top recommendations in form of urls to best matching documents
        """
        vectorized_query = self.vectorize_text([query])
        return self.get_similarities(vectorized_query).nlargest(top_n).index.to_list()

    def search_multiple(self, queries: list[str], top_n: int = 10) -> list[list[str]]:
        """
        Search recommendations for multiple queries

        :param queries: list of queries to search for
        :param top_n: number of top recommendations to return, defaults to 10
        :return: list of top recommendations in form of urls to best matching documents
        """
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
        """
        Returns a dataframe with the scores of the documents in the index for a single query.
        The scores are broken down into the scores for each token in the query.
        Last column '_SCORE_TOTAL' is the sum of the scores.

        :param query: query to search for
        :param top_n: number of top recommendations to return, defaults to 10
        :return: _description_
        """
        query_extended = np.broadcast_to(self.vectorize_text([query]), self.index.shape)
        return self.make_scores_dataframe(query_extended, top_n)

    def get_scores_dataframe_from_queries(
        self, queries: list[str], top_n: int = 10
    ) -> pd.DataFrame:
        """
        Returns a dataframe with the scores of the documents in the index for multiple queries.
        The scores are broken down into the scores for each token in the query.
        Last column '_SCORE_TOTAL' is the sum of the scores.

        :param queries: _description_
        :param top_n: number of top recommendations to return, defaults to 10
        :return: _description_
        """
        vectorized_queries = self.vectorize_text(queries)
        queries_extended = np.broadcast_to(
            vectorized_queries.mean(axis=0), self.index.shape
        )
        return self.make_scores_dataframe(queries_extended, top_n)
