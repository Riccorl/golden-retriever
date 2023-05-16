import argparse
import json
import re
from pathlib import Path
from typing import Dict

from tqdm import tqdm


def clean_annotations(input_text: str) -> Dict:
    # remove noise `BULLET::::- ` and `( )` and `List of `
    input_text = re.sub(r"BULLET::::- ", "", input_text)
    input_text = re.sub(r"\( \)", "", input_text)
    # input_text = re.sub(r"List of ", "", input_text)
    # remove double spaces
    input_text = re.sub(r"  ", " ", input_text)

    input_text = input_text.strip()

    # Find all annotations in the input text
    annotations = re.findall(r"\{ (.*?) \} \[ (.*?) \]", input_text)

    # Remove the annotations from the input text
    cleaned_text = re.sub(r"\{ (.*?) \} \[ (.*?) \]", r"@\1", input_text)

    # remove double spaces
    cleaned_text = re.sub(r"  ", " ", cleaned_text)

    doc_annotations = []
    # Find the char index of the annotated spans in the cleaned text
    for annotation in annotations:
        span_start = cleaned_text.find(f"@#@{annotation[0]}")
        cleaned_text = cleaned_text.replace("@#@", "", 1)
        span_end = span_start + len(annotation[0])
        if annotation[0] != cleaned_text[span_start:span_end]:
            raise ValueError("Annotation mismatch")
        if "List of" not in annotation[1]:
            doc_annotations.append([span_start, span_end, annotation[1]])

    text_dict = {"doc_text": cleaned_text, "doc_annotations": doc_annotations}
    return text_dict


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_file", type=str, required=True)
    arg_parser.add_argument("--output_file", type=str, required=True)
    args = arg_parser.parse_args()

    with open(args.input_file, "r", encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]

    processed_data = []
    for line_index, line in tqdm(enumerate(data)):
        processed_data.append({"doc_id": line_index, **clean_annotations(line)})

    ouput_path = Path(args.output_file)
    ouput_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ouput_path, "w", encoding="utf8") as f:
        for line in processed_data:
            f.write(json.dumps(line) + "\n")
