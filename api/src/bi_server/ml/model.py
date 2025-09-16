import numpy as np

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC

from bi_server.ml.state import (
    MULTIPLE_SPACES_PATTERN,
    SPECIAL_CHARS_PATTERN,
    SPANISH_STOPWORDS,
)


def clean_text(x: np.ndarray) -> np.ndarray:
    def clean_single_text(text: str) -> str:
        text = text.lower().strip("\"'")
        text = SPECIAL_CHARS_PATTERN.sub(" ", text)
        text = MULTIPLE_SPACES_PATTERN.sub(" ", text)

        if not text.strip():
            return ""

        tokens = word_tokenize(text, language="spanish")
        tokens = [
            token
            for token in tokens
            if token not in SPANISH_STOPWORDS and len(token) > 2
        ]
        return " ".join(tokens)

    vectorized_clean = np.vectorize(clean_single_text)
    return vectorized_clean(x)


def create_model() -> Pipeline:
    return Pipeline(
        [
            ("text_cleaner", FunctionTransformer(clean_text)),
            (
                "tfidf_vectorizer",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            (
                "svm_classifier",
                SVC(
                    kernel="linear",
                    C=1.0,
                    random_state=42,
                    probability=True,
                ),
            ),
        ]
    )
