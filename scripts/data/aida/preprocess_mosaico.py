import argparse
import json
import os
from pathlib import Path
import re
from typing import List, Tuple, Union

from ipa.preprocessing.tokenizers.spacy_tokenizer import SpacyTokenizer
from ipa.preprocessing.tokenizers.base_tokenizer import BaseTokenizer
from ipa.preprocessing.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from tqdm import tqdm


def tokenize(
    tokenizer: BaseTokenizer, document: str
) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokenized_document = tokenizer(document)
    tokens = []
    tokens_char_mapping = []
    for token in tokenized_document:
        tokens.append(token.text)
        tokens_char_mapping.append((token.start_char, token.end_char))
    return tokens, tokens_char_mapping


def split_document_by_window(
    tokenizer: BaseTokenizer,
    document: str,
    window_size: str,
    stride: int,
    doc_id: int = 0,
    doc_topic: str = None,
) -> List[dict]:
    try:
        document_tokens, tokens_char_mapping = tokenize(tokenizer, document)
    except Exception as e:
        print(document)
        raise e

    # add the first token of the document as the doc_topic
    if doc_topic is None:
        doc_topic = document_tokens[0]

    document_windows = []
    if len(document_tokens) <= window_size:
        text = document
        document_windows.append(
            dict(
                doc_id=doc_id,
                window_id=0,
                text=text,
                tokens=document_tokens,
                doc_topic=doc_topic,
                offset=0,
                token2char_start={
                    i: tokens_char_mapping[i][0] for i in range(len(document_tokens))
                },
                token2char_end={
                    i: tokens_char_mapping[i][1] for i in range(len(document_tokens))
                },
            )
        )
    else:
        for window_id, i in enumerate(range(0, len(document_tokens), stride)):
            # if the last stride is smaller than the window size, then we can
            # include more tokens form the previous window.
            if i != 0 and i + window_size > len(document_tokens):
                overflowing_tokens = i + window_size - len(document_tokens)
                if overflowing_tokens >= stride:
                    break
                i -= overflowing_tokens

            involved_token_indices = list(
                range(i, min(i + window_size, len(document_tokens) - 1))
            )

            window_tokens = [document_tokens[j] for j in involved_token_indices]
            window_text_start = tokens_char_mapping[involved_token_indices[0]][0]
            window_text_end = tokens_char_mapping[involved_token_indices[-1]][1]
            text = document[window_text_start:window_text_end]

            document_windows.append(
                dict(
                    doc_id=doc_id,
                    window_id=window_id,
                    text=text,
                    tokens=window_tokens,
                    doc_topic=doc_topic,
                    offset=window_text_start,
                    token2char_start={
                        i: tokens_char_mapping[ti][0]
                        for i, ti in enumerate(involved_token_indices)
                    },
                    token2char_end={
                        i: tokens_char_mapping[ti][1]
                        for i, ti in enumerate(involved_token_indices)
                    },
                )
            )
    return document_windows


def preprocess(
    input_file_path: Union[str, os.PathLike],
    output_file_path: Union[str, os.PathLike],
    window_size: int = 32,
    window_stride: int = 16,
    title_mapping: str = None,
    language: str = "en",
    tokenizer_device: str = "cpu",
    split_on_spaces: bool = False,
):
    if split_on_spaces:
        tokenizer = WhitespaceTokenizer()
    else:
        tokenizer = SpacyTokenizer(
            language=language,
            use_gpu=bool(tokenizer_device != "cpu"),
        )

    if title_mapping is not None:
        with open(title_mapping) as f:
            title_mapping = json.load(f)

    data = []
    with open(input_file_path) as f:
        data = [json.loads(line) for line in f]

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    num_windows = 0
    with open(output_file_path, "w") as f:
        for doc_id, document in tqdm(enumerate(data), desc="Windowizing documents"):
            # doc_id = int(document["doc_id"])
            doc_topic = None

            if len(document["entities"]) == 0:
                # no entities in the document, we skip it
                continue

            windowized_document = split_document_by_window(
                tokenizer=tokenizer,
                document=document["text"],
                window_size=window_size,
                stride=window_stride,
                doc_id=doc_id,
                doc_topic=doc_topic,
            )

            # we need to add the labels
            doc_level_labels = document["entities"]
            # if we have a title mapping, we need to map the labels to the new titles
            if title_mapping is not None:
                doc_level_labels = [
                    [start, end, title_mapping.get(label, label)]
                    for label, start, end in doc_level_labels
                ]

            # these are the labels for the whole document, we need add them to the correct window
            for window in windowized_document:
                window_level_labels = []
                for doc_level_label in doc_level_labels:
                    start_char, end_char, label_text = doc_level_label
                    if start_char >= window["offset"] and end_char <= window[
                        "offset"
                    ] + len(window["text"]):
                        window_level_labels.append(doc_level_label)
                window["window_labels"] = window_level_labels

            for window in windowized_document:
                num_windows += 1
                f.write(json.dumps(window) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_file_path", type=str, required=True)
    arg_parser.add_argument("--output_file_path", type=str, required=True)
    arg_parser.add_argument("--window_size", type=int, default=32)
    arg_parser.add_argument("--window_stride", type=int, default=16)
    arg_parser.add_argument("--title_mapping", type=str)
    arg_parser.add_argument("--language", type=str, default="en")
    arg_parser.add_argument("--tokenizer_device", type=str, default="cpu")
    arg_parser.add_argument("--split_on_spaces", action="store_true")

    preprocess(**vars(arg_parser.parse_args()))
