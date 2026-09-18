"""
scraper.py
----------
Refactored version of the project's original corpus-harvesting notebook
cells. Crawls category ("گٹھ") hub pages on `pnb.wikipedia.org` (Western
Punjabi / Shahmukhi Wikipedia) for five topical categories and saves the
resulting article texts to `data/raw/`.

This is intentionally the SAME approach as the original notebook (category
hub crawl -> article text extraction), just moved into a reusable, testable
module instead of a one-shot notebook cell, with:
  * a CLI entrypoint (`python -m src.data_collection.scraper`)
  * configurable category hubs and output path
  * basic retry/backoff instead of silently failing on transient errors
  * respectful rate limiting (`REQUEST_DELAY_SECONDS`)

License / rights note: pnb.wikipedia.org content is CC BY-SA 4.0 / GFDL.
Attribute accordingly and check current site terms before scraping at scale.

NOTE: this script needs outbound internet access to `pnb.wikipedia.org`,
which the sandboxed environment used to assemble this repo does not have
(see the project README for the domain allowlist used during development).
Run it from your own machine or a Colab/Kaggle notebook.
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3

TARGET_HUBS: dict[str, list[str]] = {
    "politics": [
        "https://pnb.wikipedia.org/wiki/گٹھ:سیاست",
        "https://pnb.wikipedia.org/wiki/گٹھ:پاکستان_دی_سیاست",
    ],
    "sports": [
        "https://pnb.wikipedia.org/wiki/گٹھ:کھیڈاں",
        "https://pnb.wikipedia.org/wiki/گٹھ:کرکٹ",
    ],
    "entertainment": [
        "https://pnb.wikipedia.org/wiki/گٹھ:پاکستانی_فلماں",
        "https://pnb.wikipedia.org/wiki/گٹھ:پنجابی_فلماں",
    ],
    "history": [
        "https://pnb.wikipedia.org/wiki/گٹھ:تریخ",
        "https://pnb.wikipedia.org/wiki/گٹھ:پنجاب_دی_تریخ",
    ],
    "geography": [
        "https://pnb.wikipedia.org/wiki/گٹھ:جغرافیہ",
        "https://pnb.wikipedia.org/wiki/گٹھ:پاکستان_دا_جغرافیہ",
    ],
}


def _get(url: str) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  retry {attempt+1}/{MAX_RETRIES} for {url}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    return None


def get_article_links_from_hub(hub_url: str) -> list[str]:
    resp = _get(hub_url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.select_one("#mw-pages") or soup
    links = []
    for a in content.select("a[href^='/wiki/']"):
        href = a.get("href", "")
        if ":" in href.split("/wiki/", 1)[-1]:
            continue  # skip Category:/Template:/Talk: pages etc.
        links.append("https://pnb.wikipedia.org" + href)
    return links


def get_article_text(article_url: str) -> str | None:
    resp = _get(article_url)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        return None
    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    text = " ".join(p for p in paragraphs if p)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def harvest(target_hubs: dict[str, list[str]] = TARGET_HUBS, min_chars: int = 80) -> pd.DataFrame:
    rows = []
    for label, hubs in target_hubs.items():
        print(f"Category: {label}")
        seen_urls: set[str] = set()
        for hub_url in hubs:
            for link in get_article_links_from_hub(hub_url):
                if link in seen_urls:
                    continue
                seen_urls.add(link)
                time.sleep(REQUEST_DELAY_SECONDS)
                text = get_article_text(link)
                if text and len(text) >= min_chars:
                    rows.append({"Text": text, "Label": label, "Source_URL": link})
        print(f"  -> {sum(1 for r in rows if r['Label'] == label)} articles collected")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = harvest()
    out_path = Path("data/raw/scraped_corpus.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
