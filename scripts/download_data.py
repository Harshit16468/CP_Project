"""
scripts/download_data.py
========================
Downloads the Natural Stories Corpus and fetches a small Wikipedia excerpt
as the background corpus for training the n-gram model.

Usage
-----
    python scripts/download_data.py

What it downloads
-----------------
  - Natural Stories text    → data/raw/natural_stories/all_stories.tok
  - Natural Stories RTs     → data/raw/natural_stories/processed_RTs.tsv
  - Wikipedia excerpt       → data/raw/ngram_corpus.txt   (~50K words)
"""

from __future__ import annotations

import logging
import shutil
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

RAW    = Path("/tmp/psycholingu/data/raw")
NS_DIR = RAW / "natural_stories"

# ---------------------------------------------------------------------------
# Natural Stories Corpus (MIT Language & Intelligence Lab)
# All files live under naturalstories_RTS/ in the repo
# ---------------------------------------------------------------------------
NS_BASE = "https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS"
NS_FILES = {
    "all_stories.tok":        f"{NS_BASE}/all_stories.tok",
    "processed_RTs.tsv":      f"{NS_BASE}/processed_RTs.tsv",
    "processed_wordinfo.tsv": f"{NS_BASE}/processed_wordinfo.tsv",
}


def download_natural_stories() -> None:
    NS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in NS_FILES.items():
        dest = NS_DIR / filename
        if dest.exists():
            logger.info("Already exists, skipping: %s", dest)
            continue
        logger.info("Downloading %s …", url)
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info("Saved to %s", dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            logger.error(
                "Please download it manually from:\n  %s\nand place it at:\n  %s",
                url, dest
            )


# ---------------------------------------------------------------------------
# Background corpus for n-gram training (Wikipedia via Wikipedia API)
# ---------------------------------------------------------------------------

def download_ngram_corpus(n_articles: int = 100) -> None:
    """
    Fetch ~n_articles Wikipedia article summaries and write to
    data/raw/ngram_corpus.txt.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "ngram_corpus.txt"
    if dest.exists():
        logger.info("N-gram corpus already exists at %s", dest)
        return

    try:
        import wikipedia  # pip install wikipedia
    except ImportError:
        logger.warning(
            "'wikipedia' package not installed. "
            "Run:  pip install wikipedia\n"
            "Then re-run this script to build the n-gram background corpus."
        )
        _write_stub_corpus(dest)
        return

    import wikipedia

    seed_topics = [
        "Linguistics", "Cognitive science", "Psycholinguistics",
        "Natural language processing", "Reading (process)", "Memory",
        "Syntax", "Semantics", "Grammar", "Neuroscience",
    ]
    texts: list[str] = []
    fetched = 0
    for topic in seed_topics:
        try:
            results = wikipedia.search(topic, results=n_articles // len(seed_topics) + 2)
            for title in results:
                if fetched >= n_articles:
                    break
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    texts.append(page.content)
                    fetched += 1
                except Exception:
                    pass
        except Exception:
            pass
        if fetched >= n_articles:
            break

    corpus = "\n\n".join(texts)
    dest.write_text(corpus, encoding="utf-8")
    logger.info("Wrote n-gram corpus: %d articles, %d chars → %s",
                fetched, len(corpus), dest)


def _write_stub_corpus(dest: Path) -> None:
    """Write a tiny stub so the pipeline doesn't crash immediately."""
    stub = (
        "The cat sat on the mat. "
        "A large language model predicts the next word in a sentence. "
        "Humans read sentences word by word and their reading times reflect comprehension difficulty. "
        "Syntactic integration cost is measured by dependency length. "
    ) * 500
    dest.write_text(stub, encoding="utf-8")
    logger.info("Wrote stub n-gram corpus to %s (replace with real data!)", dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Downloading Natural Stories Corpus …")
    download_natural_stories()

    logger.info("Building n-gram background corpus …")
    download_ngram_corpus(n_articles=100)

    logger.info("Done. Data is ready in %s", RAW)
