import re
from typing import List, Union

from overrides import overrides

from goldenretriever.serve.data.objects import Word
from goldenretriever.serve.data.tokenizers.base_tokenizer import BaseTokenizer


class RegexTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that splits the text based on a simple regex.
    """

    def __init__(self):
        super(RegexTokenizer, self).__init__()
        # regex for splitting on spaces and punctuation and new lines
        # self._regex = re.compile(r"\S+|[\[\](),.!?;:\"]|\\n")
        self._regex = re.compile(
            r"\w+|\$[\d\.]+|\S+", re.UNICODE | re.MULTILINE | re.DOTALL
        )

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        **kwargs,
    ) -> List[List[Word]]:
        """
        Tokenize the input into single words by splitting using a simple regex.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from goldenretriever.serve.tokenizers.regex_tokenizer import RegexTokenizer

            >>> regex_tokenizer = RegexTokenizer()
            >>> regex_tokenizer("Mary sold the car to John.")

        """
        # check if input is batched or a single sample
        is_batched = self.check_is_batched(texts, is_split_into_words)

        if is_batched:
            tokenized = self.tokenize_batch(texts)
        else:
            tokenized = self.tokenize(texts)

        return tokenized

    @overrides
    def tokenize(self, text: Union[str, List[str]]) -> List[Word]:
        if not isinstance(text, (str, list)):
            raise ValueError(
                f"text must be either `str` or `list`, found: `{type(text)}`"
            )

        if isinstance(text, list):
            text = " ".join(text)

        # create a spacy Token for each token
        return [
            Word(t[0], i, idx=t[1], idx_end=t[2])
            for i, t in enumerate(
                (m.group(0), m.start(), m.end()) for m in self._regex.finditer(text)
            )
        ]
