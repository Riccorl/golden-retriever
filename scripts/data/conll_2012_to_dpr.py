import json
import os
from pathlib import Path
from typing import Union, Dict, List, Optional, Any

from nlp_dataset_readers import Conll2012Reader


def conll_2012_to_dpr(
    conll_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    definitions_path: Optional[Union[str, os.PathLike]] = None,
) -> List[Dict[str, Any]]:
    # Read CoNLL 2012 file
    with open(conll_path, "r") as f:
        conll_data = json.load(f)
    # conll_data = Conll2012Reader().read(conll_path)
    # all_predicates = set(
    #     [p.sense for sentence in conll_data for p in sentence.predicates]
    # )
    definitions = {}
    if definitions_path is not None:
        # Read definitions
        with open(definitions_path, "r") as f:
            definitions = json.load(f)
    # Output DPR file, a list of dictionaries with the following keys:
    # "question": "....",
    # "answers": ["...", "...", "..."],
    # "positive_ctxs": [{
    #     "title": "...",
    #     "text": "...."
    # }],
    # "negative_ctxs": ["..."],
    # "hard_negative_ctxs": ["..."]
    dpr = []
    for sentence_id, sentence in conll_data.items():
        question = " ".join(sentence["words"])
        positive_ctxs = []
        for predicate_index, annotation in sentence["annotations"].items():
            if annotation["predicate"] not in definitions:
                print(f"Predicate {annotation['predicate']} not found in definitions")
                continue
            predicate_definition = definitions[annotation["predicate"]]
            positive_ctxs.append(
                {
                    "title": annotation["predicate"],
                    "text": predicate_definition["definition"],
                    "passage_id": f"{sentence_id}_{predicate_index}",
                }
            )
            for role_index, role in enumerate(annotation["roles"]):
                if role == "B-V" or role == "I-V" or role.startswith("I-") or role == "_":
                    continue
                role = role[2:]
                positive_ctxs.append(
                    {
                        "title": role,
                        "text": predicate_definition["roleset"].get(role, role),
                        "passage_id": f"{sentence_id}_{predicate_index}_{role_index}",
                    }
                )
        if len(positive_ctxs) == 0:
            continue
        dpr.append(
            {
                "id": f"{sentence_id}",
                "question": question,
                "answers": "",
                "positive_ctxs": positive_ctxs,
                "negative_ctxs": "",
                "hard_negative_ctxs": "",
            }
        )
    # Write DPR file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dpr, f, indent=2)
    return dpr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to CoNLL 2012 file")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument("--definitions", type=str, help="Path to output file")
    args = parser.parse_args()

    # Convert to DPR
    conll_2012_to_dpr(args.input, args.output, args.definitions)
