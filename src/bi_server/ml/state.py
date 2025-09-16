from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords

SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")

SPANISH_STOPWORDS: set[str] = set()


def nltk_required_download(root: Path):
    nltk_data_path = root / "nltk"

    nltk.download("punkt_tab", download_dir=nltk_data_path)
    nltk.download("stopwords", download_dir=nltk_data_path)

    global SPANISH_STOPWORDS
    SPANISH_STOPWORDS.clear()
    SPANISH_STOPWORDS.update(stopwords.words("spanish"))
