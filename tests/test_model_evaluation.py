import pytest

from textSummarizer.components.model_evaluation import compute_rouge, split_sentences
from textSummarizer.utils.common import clean_summary


def test_clean_summary_replaces_pegasus_line_breaks() -> None:
    assert clean_summary("Amanda baked cookies.<n>Jerry will get some.") == "Amanda baked cookies. Jerry will get some."
    assert clean_summary("  extra   spaces <n><n> here ") == "extra spaces here"


def test_split_sentences_puts_one_sentence_per_line() -> None:
    assert split_sentences("Tom is late. Is Anna coming? Yes!") == "Tom is late.\nIs Anna coming?\nYes!"
    assert split_sentences("No final punctuation") == "No final punctuation"


def test_rouge_is_perfect_for_identical_summaries() -> None:
    texts = ["Amanda baked cookies and will bring Jerry some tomorrow."]

    scores = compute_rouge(texts, texts)

    assert set(scores) == {"rouge1", "rouge2", "rougeL", "rougeLsum"}
    assert all(score == pytest.approx(1.0) for score in scores.values())


def test_rouge_is_zero_without_shared_words() -> None:
    scores = compute_rouge(["Purple trousers fit well."], ["Amanda baked cookies."])

    assert all(score == 0.0 for score in scores.values())


def test_rouge_stemming_matches_word_forms() -> None:
    # With stemming, "bakes"/"baked" and "cookie"/"cookies" count as the same words.
    scores = compute_rouge(["Amanda bakes a cookie"], ["Amanda baked a cookies"])

    assert scores["rouge1"] == pytest.approx(1.0)
