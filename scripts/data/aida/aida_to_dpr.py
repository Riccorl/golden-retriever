import json
import os
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
from transformers import AutoTokenizer, BertTokenizer


def aida_to_dpr(
    conll_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    definitions_path: Optional[Union[str, os.PathLike]] = None,
    title_map: Optional[Union[str, os.PathLike]] = None,
) -> List[Dict[str, Any]]:
    # Read AIDA file
    aida_data = []
    with open(conll_path, "r") as f:
        # aida_data = json.load(f)
        for line in f:
            aida_data.append(json.loads(line))

    definitions = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # read entities definitions
    # with open("data/aida_dpr/definitions.txt", "w") as f:
    # with open(output_path.parent / "definitions_from_dataset.txt", "w") as f_def:
    with open(definitions_path, "r") as f:
        for line in f:
            # line_data = json.loads(line)
            title, definition = line.split(" <def> ")
            title = title.strip()
            definition = definition.strip()
            definitions[title] = definition
                # f_def.write(line_data["title"] + " <def> " + line_data["text"] + "\n")
            # tokenizer.decode(line_data["text_ids"])
            # definitions[line_data["title"]].replace("[unused2]", ": ")

    if title_map is not None:
        with open(title_map, "r") as f:
            title_map = json.load(f)

    dpr = []

    for sentence in aida_data:
        question = sentence["text"]
        positive_ctxs = []
        for idx, entity in enumerate(sentence["window_labels"]):
            entity = entity[2]
            if title_map and entity in title_map:
                entity = title_map[entity]
            if entity in definitions:
                def_text = definitions[entity]
                # def_text = tokenizer.decode(definitions[entity])
                # def_text = def_text.replace("[unused2]", ": ").replace("[CLS]", "").replace("[SEP]", "")
                positive_ctxs.append(
                    {
                        "title": entity,
                        "text": f"{entity} <def> {def_text}",
                        "passage_id": f"{sentence['doc_id']}_{sentence['offset']}_{idx}",
                    }
                )

        if len(positive_ctxs) == 0:
            continue

        dpr.append(
            {
                "id": f"{sentence['doc_id']}_{sentence['offset']}",
                "question": question,
                "answers": "",
                "positive_ctxs": positive_ctxs,
                "negative_ctxs": "",
                "hard_negative_ctxs": "",
            }
        )

    # Write DPR file

    with open(output_path, "w") as f:
        json.dump(dpr, f, indent=2)

    return dpr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to AIDA file")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument(
        "--definitions", type=str, help="Path to entities definitions file"
    )
    parser.add_argument("--title_map", type=str, help="Path to title map file")
    args = parser.parse_args()

    # Convert to DPR
    aida_to_dpr(args.input, args.output, args.definitions, args.title_map)
