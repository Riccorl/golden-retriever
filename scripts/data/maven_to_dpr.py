import json
import os
from pathlib import Path
from typing import Union, Dict, List, Optional, Any


def maven_to_dpr(
    maven_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    definitions_path: Optional[Union[str, os.PathLike]] = None,
    maven_to_framenet_path: Optional[Union[str, os.PathLike]] = None,
) -> List[Dict[str, Any]]:
    # Read MAVEN file
    with open(maven_path, "r") as f:
        maven_data = [json.loads(line) for line in f.readlines()]

    definitions = {}
    if definitions_path is not None:
        # Read definitions
        with open(definitions_path, "r") as f:
            definitions = [l.strip() for l in f.readlines()]

    maven_to_framenet = {}
    if maven_to_framenet_path is not None:
        # Read definitions
        with open(maven_to_framenet_path, "r") as f:
            maven_to_framenet = json.load(f)

    # Output DPR file, a list of dictionaries with the following keys:
    # "question": "....",
    # "answers": ["...", "...", "..."],
    # "positive_pssgs": [{
    #     "title": "...",
    #     "text": "...."
    # }],
    # "negative_pssgs": ["..."],
    # "hard_negative_pssgs": ["..."]
    dpr = []
    for document in maven_data:
        d_idx = document["id"]
        for s_idx, sample in enumerate(document["content"]):
            question = sample["sentence"]
            positive_pssgs = []
            for event in document["events"]:
                for mention in event["mention"]:
                    if mention["sent_id"] != s_idx:
                        continue
                if event["type"] in maven_to_framenet:
                    event_type = maven_to_framenet[event["type"]]
                else:
                    if event["type"] not in definitions:
                        raise ValueError(
                            f"Event type {event['type']} not found in definitions"
                        )
                    event_type = event["type"].replace("_", " ").lower()

                positive_pssgs.append(
                    {
                        "title": f"{event['type']}",
                        "text": event_type,
                        "passage_id": f"{event['id']}",
                    }
                )

            if len(positive_pssgs) == 0:
                continue

            # remove duplicates
            positive_pssgs = list({v["text"]: v for v in positive_pssgs}.values())
            dpr.append(
                {
                    "id": f"{d_idx}_{s_idx}",
                    "question": question,
                    "answers": "",
                    "positive_pssgs": positive_pssgs,
                    "negative_pssgs": "",
                    "hard_negative_pssgs": "",
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
    parser.add_argument(
        "--maven_to_framenet",
        type=str,
        help="Path to output file",
    )
    args = parser.parse_args()

    # Convert to DPR
    maven_to_dpr(
        args.input,
        args.output,
        args.definitions,
        args.maven_to_framenet,
    )
