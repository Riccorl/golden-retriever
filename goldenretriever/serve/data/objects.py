from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Window:
    doc_id: int
    window_id: int
    text: str
    tokens: List[str]
    words: List[Word]
    doc_topic: str | None
    offset: int
    token2char_start: dict
    token2char_end: dict
    char2token_start: dict
    char2token_end: dict
    window_candidates: Optional[List[str]] = None


@dataclass
class Word:
    """
    A word representation that includes text, index in the sentence, POS tag, lemma,
    dependency relation, and similar information.

    # Parameters
    text : `str`, optional
        The text representation.
    index : `int`, optional
        The word offset in the sentence.
    lemma : `str`, optional
        The lemma of this word.
    pos : `str`, optional
        The coarse-grained part of speech of this word.
    dep : `str`, optional
        The dependency relation for this word.

    input_id : `int`, optional
        Integer representation of the word, used to pass it to a model.
    token_type_id : `int`, optional
        Token type id used by some transformers.
    attention_mask: `int`, optional
        Attention mask used by transformers, indicates to the model which tokens should
        be attended to, and which should not.
    """

    text: str
    i: int
    idx: int | None = None
    idx_end: int | None = None
    # preprocessing fields
    lemma: str | None = None
    pos: str | None = None
    dep: str | None = None
    head: int | None = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()
