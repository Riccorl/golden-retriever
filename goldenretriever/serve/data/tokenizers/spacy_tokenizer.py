import logging
from typing import Dict, List, Tuple, Union

import spacy

# from ipa.common.utils import load_spacy
from spacy.cli.download import download as spacy_download
from spacy.tokens import Doc

from goldenretriever.common.log import get_logger
from goldenretriever.serve.data.objects import Word
from goldenretriever.serve.data.tokenizers import SPACY_LANGUAGE_MAPPER
from goldenretriever.serve.data.tokenizers.base_tokenizer import BaseTokenizer

logger = get_logger(level=logging.DEBUG)

# Spacy and Stanza stuff

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool, bool], spacy.Language] = {}


def load_spacy(
    language: str,
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    split_on_spaces: bool = False,
) -> spacy.Language:
    """
    Download and load spacy model.

    Args:
        language (:obj:`str`, defaults to :obj:`en`):
            Language of the text to tokenize.
        pos_tags (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs POS tagging with spacy model.
        lemma (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs lemmatization with spacy model.
        parse (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs dependency parsing with spacy model.
        split_on_spaces (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will split by spaces without performing tokenization.

    Returns:
        :obj:`spacy.Language`: The spacy model loaded.
    """
    exclude = ["vectors", "textcat", "ner"]
    if not pos_tags:
        exclude.append("tagger")
    if not lemma:
        exclude.append("lemmatizer")
    if not parse:
        exclude.append("parser")

    # check if the model is already loaded
    # if so, there is no need to reload it
    spacy_params = (language, pos_tags, lemma, parse, split_on_spaces)
    if spacy_params not in LOADED_SPACY_MODELS:
        try:
            spacy_tagger = spacy.load(language, exclude=exclude)
        except OSError:
            logger.warning(
                "Spacy model '%s' not found. Downloading and installing.", language
            )
            spacy_download(language)
            spacy_tagger = spacy.load(language, exclude=exclude)

        # if everything is disabled, return only the tokenizer
        # for faster tokenization
        # TODO: is it really faster?
        # if len(exclude) >= 6:
        #     spacy_tagger = spacy_tagger.tokenizer
        LOADED_SPACY_MODELS[spacy_params] = spacy_tagger

    return LOADED_SPACY_MODELS[spacy_params]


class SpacyTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that uses SpaCy to tokenizer and preprocess the text. It returns :obj:`Word` objects.

    Args:
        language (:obj:`str`, optional, defaults to :obj:`en`):
            Language of the text to tokenize.
        return_pos_tags (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs POS tagging with spacy model.
        return_lemmas (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs lemmatization with spacy model.
        return_deps (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs dependency parsing with spacy model.
        split_on_spaces (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will split by spaces without performing tokenization.
        use_gpu (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will load the Stanza model on GPU.
    """

    def __init__(
        self,
        language: str = "en",
        return_pos_tags: bool = False,
        return_lemmas: bool = False,
        return_deps: bool = False,
        split_on_spaces: bool = False,
        use_gpu: bool = False,
    ):
        super(SpacyTokenizer, self).__init__()
        if language not in SPACY_LANGUAGE_MAPPER:
            raise ValueError(
                f"`{language}` language not supported. The supported "
                f"languages are: {list(SPACY_LANGUAGE_MAPPER.keys())}."
            )
        if use_gpu:
            # load the model on GPU
            # if the GPU is not available or not correctly configured,
            # it will rise an error
            spacy.require_gpu()
        self.spacy = load_spacy(
            SPACY_LANGUAGE_MAPPER[language],
            return_pos_tags,
            return_lemmas,
            return_deps,
            split_on_spaces,
        )
        self.split_on_spaces = split_on_spaces

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        **kwargs,
    ) -> Union[List[Word], List[List[Word]]]:
        """
        Tokenize the input into single words using SpaCy models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from ipa import SpacyTokenizer

            >>> spacy_tokenizer = SpacyTokenizer(language="en", pos_tags=True, lemma=True)
            >>> spacy_tokenizer("Mary sold the car to John.")

        """
        # check if input is batched or a single sample
        is_batched = self.check_is_batched(texts, is_split_into_words)
        if is_batched:
            tokenized = self.tokenize_batch(texts)
        else:
            tokenized = self.tokenize(texts)
        return tokenized

    def tokenize(self, text: Union[str, List[str]], *args, **kwargs) -> List[Word]:
        if self.split_on_spaces:
            if isinstance(text, str):
                text = text.split(" ")
            spaces = [True] * len(text)
            text = Doc(self.spacy.vocab, words=text, spaces=spaces)
        # return self._clean_tokens(self.spacy(text))
        return self.spacy(text)

    def tokenize_batch(
        self, texts: Union[List[str], List[List[str]]], *args, **kwargs
    ) -> List[List[Word]]:
        if self.split_on_spaces:
            if isinstance(texts[0], str):
                texts = [text.split(" ") for text in texts]
            spaces = [[True] * len(text) for text in texts]
            texts = [
                Doc(self.spacy.vocab, words=text, spaces=space)
                for text, space in zip(texts, spaces)
            ]
        return list(self.spacy.pipe(texts))
        # return [self._clean_tokens(tokens) for tokens in self.spacy.pipe(texts)]

    @staticmethod
    def _clean_tokens(tokens: Doc) -> List[Word]:
        """
        Converts spaCy tokens to :obj:`Word`.

        Args:
            tokens (:obj:`spacy.tokens.Doc`):
                Tokens from SpaCy model.

        Returns:
            :obj:`List[Word]`: The SpaCy model output converted into :obj:`Word` objects.
        """
        words = [
            Word(
                token.text,
                token.i,
                token.idx,
                token.idx + len(token),
                token.lemma_,
                token.pos_,
                token.dep_,
                token.head.i,
            )
            for token in tokens
        ]
        return words


class WhitespaceSpacyTokenizer:
    """Simple white space tokenizer for SpaCy."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        if isinstance(text, str):
            words = text.split(" ")
        elif isinstance(text, list):
            words = text
        else:
            raise ValueError(
                f"text must be either `str` or `list`, found: `{type(text)}`"
            )
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
