import platform
import string
import argparse
import asyncio
import time
import random
import math

import aiohttp
import nltk
import pandas as pd

from bs4 import BeautifulSoup
from dataclasses import dataclass

DIGIT_SET = set(string.digits)
RANDOM_WIKI_LINK = "https://en.wikipedia.org/wiki/Special:Random"
ENG_STOPWORDS = nltk.corpus.stopwords.words("english")


@dataclass
class ProgramArgs:
    number: int
    filename: str
    to: str
    batch: int
    omit_preprocess_text: bool


def parse_args() -> ProgramArgs:
    parser = argparse.ArgumentParser(
        prog="scrape.py",
        description="Parses a given number of random wikipedia articles and exports them to a file",
        epilog="Beware of too large numbers, as it may take a while to complete",
    )
    parser.add_argument("-n", "--number", type=int, help="number of articles to parse")
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="out",
        help="filename to export to (default: out)",
    )
    parser.add_argument(
        "-t",
        "--to",
        type=str,
        help="Export format (default: parquet)",
        choices=["parquet", "csv"],
        default="parquet",
    )
    parser.add_argument(
        "--omit-preprocess-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save raw, unprocessed text",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        help="Batch size for async requests (default: 100)",
        default=100,
    )

    return ProgramArgs(**vars(parser.parse_args()))


def parse_wiki_article_from_document(
    document: str | None,
) -> tuple[str, str] | tuple[None, None]:
    if not document:
        return None, None
    parsed = BeautifulSoup(document, "html.parser")
    article = parsed.find(id="content")
    if not article:
        return None, None
    title = parsed.find(id="firstHeading")
    if not title:
        return None, None
    title = title.get_text()
    text = parsed.find(id="mw-content-text")
    if not text:
        return None, None
    text = text.get_text()
    return (title, text)


class FetchException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


async def fetch_text_and_response_url(
    session: aiohttp.ClientSession, url: str
) -> tuple[str, str]:
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None, None
            return await response.text(), str(response.url)
    except aiohttp.client_exceptions.ClientConnectorError:
        return None, None


async def fetch_wiki_pages(count: int) -> list[str]:
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                fetch_text_and_response_url(session, RANDOM_WIKI_LINK)
                for _ in range(count)
            ],
            return_exceptions=False,
        )
        return results


def preprocess_raw_text(text: str | None) -> list[str]:
    if not text:
        return []
    word_lemmatizer = nltk.stem.WordNetLemmatizer()
    return [
        word_lemmatizer.lemmatize(token).lower()
        for token in nltk.word_tokenize(text)
        if not (DIGIT_SET & set(token))
        and len(token) > 1
        and token.isascii()
        and token.lower() not in ENG_STOPWORDS
    ]


def main() -> None:
    ts = time.perf_counter()
    args = parse_args()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    all_urls = []
    all_titles = []
    all_texts = []
    total = args.number
    batch_counter = 0
    total_batches = math.ceil(total / args.batch)
    while total > 0:
        doc_count = total if total < args.batch else args.batch
        print(f"Querying in batch {doc_count} documents...")
        documents, urls = zip(*asyncio.run(fetch_wiki_pages(doc_count)))
        titles, texts = zip(
            *[parse_wiki_article_from_document(document) for document in documents]
        )
        all_urls.extend(urls)
        all_titles.extend(titles)
        all_texts.extend(texts)
        total -= doc_count
        batch_counter += 1
        print(f"Finished batch {batch_counter}/{total_batches}")
        sleep_time = random.randint(1, 5)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    print(f"Successfully parsed {len(all_texts)} articles!")
    if not args.omit_preprocess_text:
        print("Pre-processing text...")
        all_texts = [";".join(preprocess_raw_text(text)) for text in all_texts]

    print("Formatting the dataframe...")
    df = pd.DataFrame(
        zip(all_titles, all_urls, all_texts), columns=["title", "url", "text"]
    )
    dropped_df = df.dropna()
    if dropped_df.shape != df.shape:
        print(
            f"Dropped {df.shape[0] - dropped_df.shape[0]} rows with missing values ({dropped_df.shape[0]} remain)"
        )
    print("Exporting to file...")
    if args.to == "parquet":
        dropped_df.to_parquet(f"{args.filename}.pq")
    elif args.to == "csv":
        dropped_df.to_csv(f"{args.filename}.csv")

    print(f"Done! Time taken: {time.perf_counter() - ts:.4f}s")


if __name__ == "__main__":
    main()
