import platform
import string
import argparse
import asyncio
import time
import random
import requests
import math

import aiohttp
import nltk
import pandas as pd

from bs4 import BeautifulSoup
from dataclasses import dataclass

DIGIT_SET = set(string.digits)
BASE_WIKI_LINK = "https://en.wikipedia.org"
RANDOM_WIKI_LINK = f"{BASE_WIKI_LINK}/wiki/Special:Random"
ENG_STOPWORDS = nltk.corpus.stopwords.words("english")


@dataclass
class ProgramArgs:
    number: int
    filename: str
    to: str
    batch: int
    omit_preprocess_text: bool
    seeding_page: str | None


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
    parser.add_argument(
        "-s",
        "--seeding-page",
        type=str,
        help="Seed the random page generator with a given page (default: None)",
        default=None,
    )

    return ProgramArgs(**vars(parser.parse_args()))


def parse_content_from_bs(
    bs: BeautifulSoup,
) -> tuple[str, str] | tuple[None, None]:
    article = bs.find(id="content")
    if not article:
        return None, None
    title = bs.find(id="firstHeading")
    if not title:
        return None, None
    title = title.get_text()
    text = bs.find(id="mw-content-text")
    if not text:
        return None, None
    text = text.get_text()
    return (title, text)


def find_links_from_bs(bs: BeautifulSoup) -> list[str]:
    links = bs.find(id="mw-content-text").find_all("a")
    return list(
        {
            BASE_WIKI_LINK + link.get("href")
            for link in links
            if link.get("href") and "wiki" in link.get("href")
        }
    )


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


def fetch_from_source(
    source: str, all_urls: list[str], all_titles: list[str], all_texts: list[str]
) -> list[str]:
    response = requests.get(source)
    if response.status_code != 200:
        raise FetchException(f"Failed to fetch from {source}")
    parsed = BeautifulSoup(response.text, "html.parser")
    title, text = parse_content_from_bs(parsed)
    all_urls.append(source)
    all_titles.append(title)
    all_texts.append(text)
    return find_links_from_bs(parsed)


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


def fetch_random_pages(
    total: int, batch: int, verbose: bool = True
) -> tuple[list[str], list[str], list[str]]:
    all_urls = []
    all_titles = []
    all_texts = []
    batch_counter = 0
    total_batches = math.ceil(total / batch)
    while total > 0:
        doc_count = total if total < batch else batch
        print(f"Querying in batch {doc_count} documents...")
        documents, urls = zip(*asyncio.run(fetch_wiki_pages(doc_count)))
        titles, texts = zip(
            *[
                parse_content_from_bs(BeautifulSoup(document, "html.parser"))
                for document in documents
                if document
            ]
        )
        all_urls.extend(urls)
        all_titles.extend(titles)
        all_texts.extend(texts)
        total -= doc_count
        batch_counter += 1
        sleep_time = random.randint(1, 5)
        if verbose:
            print(f"Finished batch {batch_counter}/{total_batches}")
            print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    return (all_urls, all_titles, all_texts)


def fetch_sync_bfs(
    source: str, total: int, verbose: bool = True
) -> tuple[list[str], list[str], list[str]]:
    all_urls = []
    all_texts = []
    all_titles = []
    all_links = [source]
    while len(all_texts) < total:
        source = all_links.pop(0)
        if verbose:
            print(f"[{len(all_texts)}/{total}] Fetching {source}")
        try:
            all_links.extend(fetch_from_source(source, all_urls, all_titles, all_texts))
        except FetchException as e:
            print(e)
            continue
        except requests.exceptions.ConnectionError as e:
            print(e)
            continue
    return (all_urls, all_titles, all_texts)


def main() -> None:
    ts = time.perf_counter()
    args = parse_args()

    if args.seeding_page:
        all_urls, all_titles, all_texts = fetch_sync_bfs(
            args.seeding_page, args.number, True
        )
    else:
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        all_urls, all_titles, all_texts = fetch_random_pages(args.number, args.batch)

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
