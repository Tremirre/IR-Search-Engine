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

LETTER_SET = set(string.ascii_lowercase)
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
    text_container = bs.find("div", {"id": "mw-content-text"})
    if not text_container:
        return None, None
    text = ""
    for paragraph in text_container.findChildren("p"):
        text += paragraph.get_text()
    return (title, text)


def find_links_from_bs(bs: BeautifulSoup) -> list[str]:
    links = bs.find(id="mw-content-text").find_all("a")
    return list(
        {
            BASE_WIKI_LINK + link.get("href")
            for link in links
            if link.get("href")
            and "wiki" in link.get("href")
            and ":" not in link.get("href")
            and "#" not in link.get("href")
            and "upload.wikimedia.org" not in link.get("href")
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
                print(
                    f"[WARNING] Failed to fetch {response.url} (received code {response.status})"
                )
                return None, None
            return await response.text(), str(response.url)
    except aiohttp.client_exceptions.ClientConnectorError:
        print(f"[WARNING] Failed to connect to the {url}")
        return None, None


async def fetch_wiki_pages(pages_to_visit: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[fetch_text_and_response_url(session, page) for page in pages_to_visit],
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
        if not (set(token.lower()) - LETTER_SET)
        and len(token) > 1
        and token.lower() not in ENG_STOPWORDS
    ]


def fetch_pages_from_list(
    batch: int, pages_to_visit: list[str], verbose: bool = True
) -> tuple[list[str], list[str], list[str]]:
    all_urls = []
    all_titles = []
    all_texts = []
    batch_counter = 0
    total_batches = math.ceil(len(pages_to_visit) / batch)
    while pages_to_visit:
        batch_pages = pages_to_visit[:batch]
        print(f"Querying in batch {len(batch_pages)} documents...")
        documents, urls = zip(*asyncio.run(fetch_wiki_pages(batch_pages)))
        titles, texts, urls = zip(
            *[
                (*parse_content_from_bs(BeautifulSoup(document, "html.parser")), url)
                for document, url in zip(documents, urls)
                if document
            ]
        )
        pages_to_visit = pages_to_visit[batch:]
        all_urls.extend(urls)
        all_titles.extend(titles)
        all_texts.extend(texts)
        batch_counter += 1
        sleep_time = random.randint(1, 5)
        if verbose:
            print(f"Finished batch {batch_counter}/{total_batches}")
            print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    return (all_urls, all_titles, all_texts)


def get_pages_related_to_source(
    source: str, total: int, verbose: bool = True
) -> list[str]:
    all_links = [source]
    parsed_links = []
    while len(all_links) < total:
        source = all_links.pop(0)
        response = requests.get(source)
        if response.status_code != 200:
            print(f"Failed to fetch {source}!")
            continue
        if verbose:
            print(f"[{len(all_links)}/{total}] Fetching links from {source}")
        new_links = find_links_from_bs(BeautifulSoup(response.text, "html.parser"))
        new_links = [
            link
            for link in new_links
            if link not in parsed_links + all_links + [source]
        ]
        all_links.extend(new_links)
        parsed_links.append(source)
    return (parsed_links + all_links)[:total]


def main() -> None:
    ts = time.perf_counter()
    args = parse_args()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    pages = [RANDOM_WIKI_LINK] * args.number
    if args.seeding_page:
        pages = get_pages_related_to_source(args.seeding_page, args.number)

    all_urls, all_titles, all_texts = fetch_pages_from_list(args.batch, pages)

    print(f"Successfully parsed {len(all_texts)}/{args.number} articles!")

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
            f"Dropped {df.shape[0] - dropped_df.shape[0]} rows with missing values ({dropped_df.shape[0]} remain)."
        )
    print("Exporting to file...")
    if args.to == "parquet":
        dropped_df.to_parquet(f"{args.filename}.pq")
    elif args.to == "csv":
        dropped_df.to_csv(f"{args.filename}.csv")

    print(f"Done! Time taken: {time.perf_counter() - ts:.4f}s")


if __name__ == "__main__":
    main()
