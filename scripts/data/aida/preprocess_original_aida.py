import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# from ipa.preprocessing.tokenizers.stanza_tokenizer import StanzaTokenizer
# from ipa.preprocessing.tokenizers.spacy_tokenizer import SpacyTokenizer
from goldenretriever.serve.tokenizers import WhitespaceTokenizer, SpacyTokenizer, RegexTokenizer
from goldenretriever.serve.tokenizers.base_tokenizer import BaseTokenizer

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
    document_tokens, tokens_char_mapping = tokenize(tokenizer, document)
    if doc_topic is None:
        doc_topic = document_tokens[0]
    document_windows = []
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
    output_dir_path: Union[str, os.PathLike],
    window_size: int = 32,
    window_stride: int = 16,
    title_mapping: str = None,
    language: str = "en",
    tokenizer_device: str = "cpu",
    split_on_spaces: bool = False,
):
    if split_on_spaces:
        # tokenizer = WhitespaceTokenizer()
        tokenizer = RegexTokenizer()
    else:
        tokenizer = SpacyTokenizer(
            language=language,
            use_gpu=bool(tokenizer_device != "cpu"),
            # split_on_spaces=split_on_spaces,
        )

    if title_mapping is not None:
        with open(title_mapping) as f:
            title_mapping = json.load(f)

    data = []
    with open(input_file_path) as f:
        for line in f:
            data.append(json.loads(line))

    # missing labels for debugging
    # missing_labels = set()

    windowized_data_train = []
    windowized_data_dev = []
    windowized_data_test = []
    for document in tqdm(data, desc="Windowizing documents"):
        doc_info = document["doc_id"]

        # clean doc_info, e.g. "-DOCSTART- (1 EU)"
        doc_info = (
            doc_info.replace("-DOCSTART-", "").replace("(", "").replace(")", "").strip()
        )
        doc_id, doc_topic = doc_info.split(" ")

        if "testa" in doc_id:
            split = "dev"
        elif "testb" in doc_id:
            split = "test"
        else:
            split = "train"

        doc_id = doc_id.replace("testa", "").replace("testb", "").strip()
        doc_id = int(doc_id)

        windowized_document = split_document_by_window(
            tokenizer=tokenizer,
            document=document["doc_text"],
            window_size=window_size,
            stride=window_stride,
            doc_id=doc_id,
            doc_topic=doc_topic,
        )

        # we need to add the labels
        doc_level_labels = document["doc_annotations"]
        # if we have a title mapping, we need to map the labels to the
        # new titles
        if title_mapping is not None:
            # compute the missing labels
            # missing_labels |= set(title_mapping.keys()) - set(
            #     [label for _, _, label in doc_level_labels]
            # )
            doc_level_labels = [
                [start, end, title_mapping.get(label, label)]
                for start, end, label in doc_level_labels
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

            # now we need to map the labels to the tokens
            window_level_labels_but_for_tokens = []
            for label in window_level_labels:
                start_char, end_char, label_text = label
                start_token = None
                end_token = None
                for token_id, (start, end) in enumerate(
                    zip(
                        window["token2char_start"].values(),
                        window["token2char_end"].values(),
                    )
                ):
                    if start_char == start:
                        start_token = token_id
                    if end_char == end:
                        end_token = token_id + 1
                if start_token is None or end_token is None:
                    raise ValueError(
                        f"Could not find token for label: {label} in window: {window}"
                    )
                window_level_labels_but_for_tokens.append(
                    [start_token, end_token, label_text]
                )
            window["window_labels_tokens"] = window_level_labels_but_for_tokens

        if split == "train":
            windowized_data_train.extend(windowized_document)
        elif split == "dev":
            windowized_data_dev.extend(windowized_document)
        elif split == "test":
            windowized_data_test.extend(windowized_document)
        else:
            raise ValueError(f"Unknown split: {split}")

    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / "train_windowed.jsonl", "w") as f:
        for window in windowized_data_train:
            f.write(json.dumps(window) + "\n")
    with open(output_dir_path / "testa_windowed.jsonl", "w") as f:
        for window in windowized_data_dev:
            f.write(json.dumps(window) + "\n")
    with open(output_dir_path / "testb_windowed.jsonl", "w") as f:
        for window in windowized_data_test:
            f.write(json.dumps(window) + "\n")

    # print(f"Missing labels: {missing_labels}")
    # print(f"Total number of missing labels: {len(missing_labels)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_file_path", type=str, required=True)
    arg_parser.add_argument("--output_dir_path", type=str, required=True)
    arg_parser.add_argument("--window_size", type=int, default=32)
    arg_parser.add_argument("--window_stride", type=int, default=16)
    arg_parser.add_argument("--title_mapping", type=str)
    arg_parser.add_argument("--language", type=str, default="en")
    arg_parser.add_argument("--tokenizer_device", type=str, default="cpu")
    arg_parser.add_argument("--split_on_spaces", action="store_true")

    preprocess(**vars(arg_parser.parse_args()))
